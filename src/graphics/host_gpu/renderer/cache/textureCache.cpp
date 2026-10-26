#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/abSwitch.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/colorClearCheck.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <span>
#include <tuple>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Overlapping aliases (transient render targets sharing memory) stay registered until unused for
// this many presented frames as well; see AliasAgeByFrames.
constexpr uint64_t AliasFramesBeforeRemoval = 4;

// An overlapped alias was freed once unused for NumFramesBeforeRemoval scheduler ticks. A title
// submits hundreds of times a frame, so that is usually within the frame that used it: the
// aliases of a transient image pool (Wolverine: ~4 a frame) were destroyed and recreated from
// guest memory every frame. They now also have to be unused for AliasFramesBeforeRemoval
// presented frames, and live aliases follow a single-owner rule (CommitGpuWrite).
// KYTY_IMAGE_ALIAS_AGE=0 (or =ticks) restores the tick rule alone.
[[nodiscard]] bool AliasAgeByFrames() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_IMAGE_ALIAS_AGE");
		return value == nullptr ||
		       (std::strcmp(value, "0") != 0 && std::strcmp(value, "ticks") != 0);
	}();
	return enabled;
}

// The collector writes a GPU-written image back to guest memory before it frees it (under
// memory pressure). Whether the image is CPU-dirty is checked when the copy is recorded, but the
// bytes land only once the GPU is done with that tick, and the image is untracked by then: a
// guest that reused the memory in between (a retired render target becoming a CPU-written vertex
// table, as in Jetsku's 919e712a) had it overwritten with stale pixels. The guest bytes are hashed
// when the copy is recorded and again before it is published; if they changed, the write-back
// is dropped. KYTY_EVICT_WRITEBACK_GUARD=0 publishes unconditionally again.
[[nodiscard]] bool EvictWriteBackGuard() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_EVICT_WRITEBACK_GUARD");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Hash of the guest backing bytes of a range, read in chunks; false if a chunk is unreadable.
[[nodiscard]] bool HashGuestBacking(uint64_t address, uint64_t size, uint64_t& hash) {
	constexpr uint64_t   Chunk = 64 * 1024;
	std::vector<uint8_t> bytes(static_cast<size_t>(std::min(size, Chunk)));
	uint64_t             seed = 0;
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min(size - done, Chunk);
		if (!LibKernel::Memory::TryReadBacking(address + done, bytes.data(), part)) {
			return false;
		}
		seed = XXH3_64bits_withSeed(bytes.data(), static_cast<size_t>(part), seed);
		done += part;
	}
	hash = seed;
	return true;
}

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		if (desc.type != TextureCache::BindingType::RenderTarget || !metadata.clear_register_valid) {
			return false;
		}
		if (DecodePackedColorClear(format, metadata.clear_word, clear)) {
			return true;
		}
		// KYTY_CLEAR_REGISTER_WIDE: 64-bit, 16-bit-channel and B10G11R11 targets (both words).
		if (!DecodePackedColorClear64(format, metadata.clear_word, metadata.clear_word1, clear)) {
			return false;
		}
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint32_t> logged;
		if (std::lock_guard lock(logged_mutex); logged.insert(static_cast<uint32_t>(format)).second) {
			LOGF("Register clear of format %u decoded with both clear words (was dropped)\n",
			     static_cast<uint32_t>(format));
		}
		return true;
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

[[nodiscard]] std::vector<vk::BufferImageCopy> BuildDepthCopies(const ImageInfo& info,
                                                              uint64_t slice_stride,
                                                              vk::ImageAspectFlags aspect) {
	std::vector<vk::BufferImageCopy> copies(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		auto& copy             = copies[layer];
		copy.bufferOffset      = slice_stride * layer;
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {aspect, 0, layer, 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
	}
	return copies;
}

[[nodiscard]] std::vector<GpuTileInfo> BuildDepthTiles(const ImageInfo& info) {
	TileBlockLayout block {};
	EXIT_NOT_IMPLEMENTED(
	    !TileGetBlockLayout(TileBlockFamily::Depth64KB, info.bytes_per_block, block));
	const auto               full_slice_size = info.data.size / info.resources.layers;
	std::vector<GpuTileInfo> tiles;
	tiles.reserve(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		const auto offset = full_slice_size * layer;
		tiles.push_back({block.family, block.bytes_per_element, offset, full_slice_size, offset,
		                 full_slice_size, 0, info.extent.width, info.extent.height, 1, info.pitch});
		tiles.back().surface_z = layer;
	}
	return tiles;
}

// The first native texel holds the image's first guest element once its color transform is
// undone.
[[nodiscard]] bool HasGuestTexelLayout(const Image& image) {
	const auto& info = image.info;
	return image.registered && !image.depth_id && image.backing.image != nullptr &&
	       !info.IsDepth() && !info.IsBlock() && !info.IsVolume() && info.samples == 1 &&
	       image.backing.samples == 1 && info.mip_layout[0].offset == 0 &&
	       info.metadata.compression == VideoOutCompression::Uncompressed &&
	       info.bytes_per_block != 0 && vk::blockSize(image.backing.format) == info.bytes_per_block;
}

} // namespace

// KYTY_GPU_CLEAR_CHECK, per A/B arm (common/abSwitch.h): 0 is off, unset or 1-3 on.
static const Common::AbSwitch<bool>& ClearCheckSwitch() {
	static const Common::AbSwitch<bool> value("KYTY_GPU_CLEAR_CHECK", [](const char* v) {
		return v == nullptr || v[0] == '1' || v[0] == '2' || v[0] == '3';
	});
	return value;
}

// KYTY_TEXTURE_CACHE_MB / KYTY_BUFFER_CACHE_MB: an upper bound for a cache's collection thresholds,
// whatever the card's size (default 6 GiB each; 0 = no bound). The thresholds otherwise follow the
// device budget, and on a 32 GB card the caches kept everything up to ~25 GB: textures and buffers
// of scenes the game had left stayed until the card was full (a new game's prologue, 2026-10-07).
static uint64_t CacheCapBytes(const char* name) {
	const char* value = std::getenv(name);
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 6144ull;
	return mib == 0 ? UINT64_MAX : mib << 20u;
}

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
	if (const auto cap = CacheCapBytes("KYTY_TEXTURE_CACHE_MB"); cap != UINT64_MAX) {
		m_pressure_gc_memory = std::min(m_pressure_gc_memory, cap);
		m_critical_gc_memory = std::min(m_critical_gc_memory, cap + cap / 4);
		m_trigger_gc_memory  = std::min(m_trigger_gc_memory, cap / 4 * 3);
		LOGF("TextureCache: thresholds capped at %" PRIu64 " MB (trigger %" PRIu64 ", critical %" PRIu64
		     " MB)\n",
		     m_pressure_gc_memory >> 20u, m_trigger_gc_memory >> 20u, m_critical_gc_memory >> 20u);
	}
	// Color clear checks of GPU-written metadata run on the GPU (KYTY_GPU_CLEAR_CHECK=0: on the CPU,
	// draining the GPU). UFC 5 fight route, 2026-10-15, known fills on: the CPU check drained ~1.2
	// times a frame (~7.5 ms each, a CMask a compute pass rewrites); on the GPU, DrawAuto 155 -> 8.5
	// us a call, screenshots the same. In-run A/B (common/abSwitch.h), 2026-10-16, standing fight:
	// off is 17.8 % slower (+-0.7 %), 24.0 against 29.2 fps, 13.9 against 9.9 us a draw.
	if (ClearCheckSwitch().Get(0) || ClearCheckSwitch().Get(1)) {
		if (m_graphics.conditional_rendering_enabled) {
			m_clear_check = std::make_unique<ColorClearCheck>(graphics, scheduler);
		}
		LOGF("TextureCache: GPU color clear check %s\n",
		     m_clear_check != nullptr ? "on" : "unavailable (no VK_EXT_conditional_rendering)");
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	m_slot_images[id].self_id = id;
	Profiler::Add(Profiler::Counter::ImagesCreated);
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	// The page bits are set before TrackImage can protect the image's pages: a write fault that
	// its protection causes sees them and takes m_lock (InvalidateMemory).
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
		m_image_pages[page / 64].fetch_or(uint64_t {1} << (page % 64), std::memory_order_seq_cst);
	});
	image.registered = true;
	++m_image_set_epoch;
	InvalidateRegionQueries(image.info.data.address, image.info.data.size);
	NoteBindlessStateChange(image);
	image.lru_id     = m_lru_cache.Insert(id, LruClock());
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.bindless_pinned) {
		NoteBindlessStateChange(image);
		image.bindless_pinned = false;
		if (on_bindless_unregister) {
			on_bindless_unregister(id);
		}
	}
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
		if (owners->empty()) {
			m_image_pages[page / 64].fetch_and(~(uint64_t {1} << (page % 64)),
			                                   std::memory_order_seq_cst);
		}
	});
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
	++m_image_set_epoch;
	InvalidateRegionQueries(image.info.data.address, image.info.data.size);
}

void TextureCache::InvalidateRegionQueries(uint64_t address, uint64_t size) {
	// Only queries over the image's page-table pages can find it.
	const auto first = address >> ImagePageTable::kPageBits;
	const auto last  = (address + size - 1) >> ImagePageTable::kPageBits;
	for (auto& query: m_region_queries) {
		if (query.epoch == m_registry_epoch &&
		    (query.address >> ImagePageTable::kPageBits) <= last &&
		    ((query.address + query.size - 1) >> ImagePageTable::kPageBits) >= first) {
			query.epoch = 0;
		}
	}
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	if (!image->depth_id) {
		std::vector<ImageId> associations;
		m_slot_images.ForEach([&](ImageId candidate, const Image& associated) {
			if (associated.depth_id == id) {
				associations.push_back(candidate);
			}
		});
		for (const auto association: associations) {
			FreeImage(association);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	// Also between guest command buffers (a guest unmap runs there): command buffers submitted
	// earlier may still be sampling the image on the GPU. Destroying it at once let them read
	// freed memory (device loss at the title, 2026-10-04).
	m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
}

void TextureCache::ReleaseImage(ImageId id) {
	const auto* image = m_slot_images.try_get(id);
	if (image != nullptr && on_image_release && on_image_release(*image)) {
		// Work recorded with the repointed slot runs until the current tick is done.
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	if (image != nullptr && !m_scheduler.IsFree(image->tick_accessed_last)) {
		// Used after it was dropped, in a tick the GPU has not finished: keep it until then.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("TextureCache: image %ux%u guest=0x%016" PRIx64 " used in tick %" PRIu64
			     " after its release was queued; destruction waits for that tick\n",
			     image->info.extent.width, image->info.extent.height, image->info.data.address,
			     image->tick_accessed_last);
		}
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	m_slot_images.erase(id);
	Profiler::Add(Profiler::Counter::ImagesDestroyed);
}

void TextureCache::FreeImage(ImageId id) {
	auto& image = m_slot_images[id];
	ClearOverwrittenTexels(id);
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	image.frame_accessed_last = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, LruClock());
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	if (m_readback_linear_images && !image.info.IsTiled() && !image.info.data.Empty()) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	const auto slot =
	    ((address ^ (size << 24u) ^ static_cast<uint64_t>(page_overlap)) * 0x9e3779b97f4a7c15ull) >>
	    (64u - RegionQueryBits);
	auto& query = m_region_queries[slot];
	auto& ids   = m_region_query_ids[slot];
	if (query.epoch == m_registry_epoch && query.address == address && query.size == size &&
	    query.page_overlap == page_overlap) {
		ImageIds cached;
		for (uint32_t i = 0; i < query.count; i++) {
			cached.push_back(ids[i]);
		}
		return cached;
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	if (result.size() <= ids.size()) {
		query = {.address      = address,
		         .size         = size,
		         .epoch        = m_registry_epoch,
		         .page_overlap = page_overlap,
		         .count        = static_cast<uint8_t>(result.size())};
		for (uint32_t i = 0; i < query.count; i++) {
			ids[i] = result[i];
		}
	}
	return result;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto key = std::pair {desc.info.pixel_format, desc.info.type};
	if (const auto found = m_null_images.find(key); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = desc.info.type;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(key, id);
	return id;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	std::vector<vk::BufferImageCopy> depth_copies(layers);
	std::vector<vk::BufferImageCopy> color_copies(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		depth_copies[layer].bufferOffset      = depth_slice * layer;
		depth_copies[layer].bufferRowLength   = depth.info.pitch;
		depth_copies[layer].bufferImageHeight = depth.info.extent.height;
		depth_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, layer, 1};
		depth_copies[layer].imageExtent       = depth.info.extent;
		color_copies[layer].bufferOffset      = color_slice * layer;
		color_copies[layer].bufferRowLength   = color.info.pitch;
		color_copies[layer].bufferImageHeight = color.info.extent.height;
		color_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, layer, 1};
		color_copies[layer].imageExtent       = color.info.extent;
	}

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
		destination.gpu_write_seq = std::max(destination.gpu_write_seq, source.gpu_write_seq);
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
		destination.gpu_write_seq = std::max(destination.gpu_write_seq, source.gpu_write_seq);
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	           (cached.info.metadata.kind == ImageMetadataKind::None ||
        cached.info.metadata.kind == ImageMetadataKind::Htile);
	// PPSA04264
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			// PPSA03541
			// Dynamic resolution keeps the depth pitch and height while the width changes. An
			// attachment narrower than the request is a GPU fault (ZT violation, device lost).
			recreate |= cached.info.extent.width < requested.extent.width ||
			            cached.info.extent.height < requested.extent.height;
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout = cached.info.mip_layout;
	} else {
		info.resources = std::max(requested.resources, cached.info.resources);
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr) {
		return {merged_id};
	}
	auto&      cached       = *owner;
	const auto current_tick  = m_scheduler.CurrentTick();
	const auto current_frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    (!AliasAgeByFrames() || current_frame - std::min(current_frame, cached.frame_accessed_last) >
	                                AliasFramesBeforeRemoval);

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			// A texel alias would be recreated from stale guest bytes, so it stays.
			if (safe_to_delete && !IsTexelAlias(cached)) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		if (cached.binding.is_target && !m_slot_images[merged_id].binding.is_target) {
			NoteBindlessStateChange(m_slot_images[merged_id]);
		}
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete &&
	    !IsTexelAlias(cached)) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	return expanded_id;
}

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	ColorTransform                   color_transform = ColorTransform::None;
	bool                             valid           = false;

	[[nodiscard]] uint64_t LinearSize() const {
		uint64_t size = 0;
		for (const auto& tile: tiles) {
			size = std::max(size, tile.linear_offset + tile.linear_size);
		}
		return size;
	}
};

struct TextureCache::ImageDownload {
	TextureTransfer texture;
	bool                depth_target = false;
	bool                valid        = false;
};

TextureCache::TextureTransfer
TextureCache::BuildTextureTransfer(const Image& image, BindingType binding,
                                    TransferDirection direction) const {
	const auto& info             = image.info;
	const bool  upload           = direction == TransferDirection::Upload;
	const bool  render_target    = binding == BindingType::RenderTarget;
	const bool  video_out        = binding == BindingType::VideoOut;
	uint32_t    layers           = info.TransferLayers();
	bool        volume           = info.IsVolume();
	const char* owner            = "TextureCache readback";

	TextureTransfer transfer;
	transfer.color_transform = info.GetColorTransform();
	if (upload) {
		if ((render_target || video_out) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		owner = "TextureCache";
		if (render_target) {
			owner = "RenderTarget";
		} else if (binding == BindingType::Storage) {
			owner = "StorageTextureCache";
		} else if (video_out) {
			if (info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("TextureCache: invalid color-attachment upload\n");
			}
			layers = info.resources.layers;
			volume = false;
			owner  = "VideoOut";
		}
	}

	transfer.layout = TextureCalcUploadLayout(info.guest_format, info.extent.width,
	                                         info.extent.height, info.resources.levels, layers,
	                                         info.tile_mode, info.data.size, volume, owner);
	transfer.regions = TextureBuildImageCopies(transfer.layout);
	if (info.IsDepth()) {
		for (auto& region: transfer.regions) {
			region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
		}
	}
	if (transfer.layout.surface.description.tile_mode != Prospero::TileMode::kLinear) {
		if (!TextureBuildGpuTileInfos(info.data.size, transfer.regions, transfer.layout,
		                              info.resources.levels, transfer.tiles)) {
			return transfer;
		}
	}
	transfer.valid = true;
	return transfer;
}

TextureCache::ImageDownload TextureCache::BuildDownload(const Image& image) const {
	const auto&  info    = image.info;
	const auto   binding = UploadBinding(image);
	ImageDownload transfer {.depth_target = binding == BindingType::DepthTarget};
	if (info.samples != 1 || image.backing.samples != 1) {
		return transfer;
	}
	if (transfer.depth_target) {
		transfer.valid = IsSupportedDepthPlaneReadback(info) && info.resources.layers != 0 &&
		             info.data.size % info.resources.layers == 0 &&
		             Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
		return transfer;
	}
	if (info.metadata.compression != VideoOutCompression::Uncompressed) {
		return transfer;
	}
	transfer.texture = BuildTextureTransfer(image, binding, TransferDirection::Download);
	transfer.valid   = transfer.texture.valid;
	return transfer;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	const auto  upload  = [&](std::vector<vk::BufferImageCopy>& copies, TileManager::Result linear) {
		for (auto& copy: copies) {
			copy.bufferOffset += linear.offset;
		}
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	};

	if (binding != BindingType::DepthTarget) {
		const auto& info = image.info;
		auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		if (!transfer.valid) {
			EXIT("TextureCache: invalid texture upload: binding=%u addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " format=%u tile=%u family=%u extent=%ux%ux%u "
			     "pitch=%u levels=%u layers=%u samples=%u\n",
			     static_cast<uint32_t>(binding), info.data.address, info.data.size,
			     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
			     static_cast<uint32_t>(transfer.layout.surface.texture.block.family), info.extent.width,
			     info.extent.height, info.extent.depth, info.pitch, info.resources.levels,
			     info.resources.layers, info.samples);
		}
		TileManager::Result linear {source.Handle(), source_offset, info.data.size};
		if (!transfer.tiles.empty()) {
			linear = m_tiler.Detile(source.Handle(), source_offset, info.data.size,
			                        transfer.LinearSize(), transfer.tiles, transfer.color_transform);
		} else if (transfer.color_transform != ColorTransform::None) {
			linear = m_tiler.TransformColor(linear, transfer.color_transform, true);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.resources       = {image.stencil_subresources.level_count,
		                        image.stencil_subresources.layer_count};
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.bytes_per_block = 1;
		if (info.IsTiled()) info.pitch = TileGetDepthPitch(info.extent.width, 1, 0);
	}
	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto          layers          = info.resources.layers;
	const auto          full_slice_size = info.data.size / layers;
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	TileManager::Result linear {source.Handle(), source_offset, source.Size() - source_offset};
	if (info.IsTiled()) {
		const auto tiles = BuildDepthTiles(info);
		linear =
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles);
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		for (uint32_t layer = 0; layer < layers; layer++) {
			copies[layer].bufferOffset = transfer_slice * layer;
		}
	}
	upload(copies, linear);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range        = desc.info.metadata.range;
	const auto current_tick = m_scheduler.CurrentTick();
	const auto layers       = desc.info.TransferLayers();
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
		}
		// A fast-clear stamps the whole DCC metadata range with one uniform code; an ordinary
		// draw also touches these bytes (per-block compression state) but leaves a mixed
		// pattern, so it never passes the all-same-code check below -- yet it re-marks the range
		// GPU-dirty just the same, so gating purely on that flag re-triggers the synchronous
		// readback on every draw to an already-bound render target. Gate on the scheduler tick
		// instead: within one still-unsubmitted recording no GPU write can land in guest memory
		// after the first check, so a later FindImage for the same slices of the same image
		// this tick is redundant. Other slices of the image (another layer or face) keep their
		// own check, a CPU store to the metadata (a guest-side clear) is caught through the
		// tracker below, and a real re-clear in a later recording still gets caught, since the
		// tick will have advanced.
		const bool already_resolved_this_tick =
		    image.dcc_clear_checked_tick == current_tick &&
		    image.dcc_clear_checked_first == first && image.dcc_clear_checked_count == count;
		image.dcc_clear_checked_tick  = current_tick;
		image.dcc_clear_checked_first = first;
		image.dcc_clear_checked_count = count;
		if (already_resolved_this_tick &&
		    !m_buffer_cache.IsRegionCpuModified(range.address, range.size)) {
			return;
		}
	}
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	const auto slice_size = range.size / layers;
	// GPU-written metadata whose bytes a recorded fill still accounts for (BufferCache::
	// TryKnownFill) is read from that record, slice by slice, instead of downloading: a download
	// submits and drains the GPU. Diagnostics count both.
	static uint64_t known_slices      = 0;
	static uint64_t downloaded_checks = 0;
	std::vector<uint32_t> known_values(count, 0);
	std::vector<uint8_t>  known(count, 0);
	bool                  all_known = false;
	if (m_buffer_cache.IsRegionGpuModified(range.address, range.size) &&
	    m_buffer_cache.HasGpuDirtyBytes(range.address, range.size)) {
		all_known = true;
		for (uint32_t slice = 0; slice < count; slice++) {
			const auto address = range.address + slice_size * (first + slice);
			known[slice] = m_buffer_cache.TryKnownFill(address, slice_size, known_values[slice]) ? 1 : 0;
			all_known    = all_known && known[slice] != 0;
		}
	}
	// KYTY_KNOWN_FILLS=2 (diagnostics): the CPU check still decides; the known fills' answers are
	// compared with the downloaded bytes, and mismatches are logged.
	static const bool verify_known = [] {
		const char* value = std::getenv("KYTY_KNOWN_FILLS");
		return value != nullptr && value[0] == '2';
	}();
	std::vector<uint8_t> verify(count, 0);
	if (verify_known) {
		verify = known;
		std::fill(known.begin(), known.end(), uint8_t {0});
		all_known = false;
	}
	// GPU-written metadata no known fill accounts for: check it on the GPU instead of draining
	// (KYTY_GPU_CLEAR_CHECK=1, ColorClearCheck).
	if (!all_known && m_clear_check != nullptr && ClearCheckSwitch().Get() && !volume_texture &&
	    m_buffer_cache.IsRegionGpuModified(range.address, range.size) &&
	    m_buffer_cache.HasGpuDirtyBytes(range.address, range.size)) {
		const auto checked_address = range.address + slice_size * first;
		const auto checked_size    = slice_size * count;
		// Nothing wrote these slices since their last check (and the CPU did not either).
		if (m_buffer_cache.IsClearChecked(checked_address, checked_size) &&
		    !m_buffer_cache.IsRegionCpuModified(range.address, range.size)) {
			return;
		}
		if (MaterializeColorClearOnGpu(id, desc, range.address, slice_size, first, image_first,
		                               count)) {
			m_buffer_cache.MarkClearChecked(checked_address, checked_size);
			return;
		}
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	if (!all_known && m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		std::fill(known.begin(), known.end(), uint8_t {0});
		const bool dirty = m_buffer_cache.HasGpuDirtyBytes(range.address, range.size);
		if (dirty) {
			downloaded_checks++;
		}
		// KYTY_CLEAR_CHECK_WRITERS=1: which GPU write the drain below waits for, and how long.
		const auto reader = g_gpu_write_tag;
		uint64_t   writer = 0, writer_tick = 0;
		const bool writer_known =
		    dirty && ClearCheckWritersEnabled() &&
		    m_buffer_cache.LastGpuWriter(range.address, range.size, writer, writer_tick);
		const auto started = std::chrono::steady_clock::now();
		m_buffer_cache.ReadMemory(range.address, range.size, false);
		if (dirty && ClearCheckWritersEnabled()) {
			static uint32_t logged = 0;
			if (logged++ < 400) {
				const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
				                    std::chrono::steady_clock::now() - started)
				                    .count();
				uint32_t known_count = 0;
				for (const auto k: known) {
					known_count += k;
				}
				LOGF("Clear check drain: %" PRId64 " us addr=0x%016" PRIx64 " size=0x%" PRIx64
				     " slices=%u+%u/%u known=%u kind=%s format=%u %ux%u reader=%s 0x%016" PRIx64
				     " writer=%s%s%s 0x%016" PRIx64 " ticks_ago=%" PRId64 "\n",
				     static_cast<int64_t>(us), range.address, range.size, first, count, layers,
				     known_count,
				     desc.info.metadata.kind == ImageMetadataKind::Dcc ? "dcc" : "cmask",
				     static_cast<uint32_t>(view.format), desc.info.extent.width,
				     desc.info.extent.height, (reader & GpuWriteTagDraw) != 0 ? "draw" : "dispatch",
				     reader & 0x0000ffffffffffffull,
				     !writer_known ? "unknown" : (writer & GpuWriteTagDraw) != 0 ? "draw" : "dispatch",
				     (writer & GpuWriteTagFill) != 0 ? "+fill" : "",
				     (writer & GpuWriteTagCopy) != 0 ? "+copy" : "", writer & 0x0000ffffffffffffull,
				     writer_known ? static_cast<int64_t>(current_tick - writer_tick) : int64_t {-1});
			}
		}
	} else if (all_known) {
		known_slices += count;
	}
	if ((known_slices + downloaded_checks) % 1024 == 1 && (known_slices | downloaded_checks) != 0) {
		LOGF("TextureCache: color metadata from known fills %" PRIu64 " slices, downloaded %" PRIu64
		     " checks\n",
		     known_slices, downloaded_checks);
	}
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		std::vector<uint8_t> bytes(slice_size);
		if (known[slice] != 0) {
			for (uint64_t i = 0; i < slice_size; i++) {
				bytes[i] = static_cast<uint8_t>(known_values[slice] >> (8u * (i & 3u)));
			}
		} else if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			EXIT("TextureCache: failed to read color metadata slice\n");
		}
		if (verify[slice] != 0) {
			static uint64_t verified   = 0;
			static uint64_t mismatched = 0;
			uint64_t        wrong      = 0;
			for (uint64_t i = 0; i < slice_size; i++) {
				wrong += bytes[i] != static_cast<uint8_t>(known_values[slice] >> (8u * (i & 3u))) ? 1 : 0;
			}
			verified++;
			if (wrong != 0 && ++mismatched <= 32) {
				LOGF("KnownFills verify: MISMATCH addr=0x%016" PRIx64 " slice=%u/%u size=0x%" PRIx64
				     " known=0x%08" PRIx32 " actual[0]=0x%02x wrong_bytes=%" PRIu64 " type=%u format=%u\n",
				     address, first + slice, layers, slice_size, known_values[slice], bytes[0], wrong,
				     static_cast<uint32_t>(desc.type), static_cast<uint32_t>(desc.view_info.format));
			}
			if (verified % 1024 == 1) {
				LOGF("KnownFills verify: %" PRIu64 " slices, %" PRIu64 " mismatched\n", verified,
				     mismatched);
			}
		}
		const uint8_t  code = bytes[0];
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			continue;
		}
		{
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Known metadata is still GPU-written: the expanded keys go to the GPU copy, in order,
		// and are recorded, so the next check reads them without a download either.
		if (known[slice] != 0) {
			// KYTY_KNOWN_FILLS=3: the expanded keys reach guest memory too, as after the CPU
			// check (WriteClean: no download, and the bytes stop being GPU-written).
			static const bool write_clean = [] {
				const char* value = std::getenv("KYTY_KNOWN_FILLS");
				return value != nullptr && value[0] == '3';
			}();
			if (desc.type != BindingType::VideoOut) {
				const std::vector<uint8_t> keys(slice_size, uint8_t {0xff});
				if (!write_clean || !m_buffer_cache.WriteClean(address, keys.data(), keys.size())) {
					m_buffer_cache.FillKnown(address, slice_size, 0xffffffffu);
				}
			}
			continue;
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
	}
}

bool TextureCache::MaterializeColorClearOnGpu(ImageId id, const ImageDesc& desc, uint64_t address,
                                              uint64_t slice_size, uint32_t first,
                                              uint32_t image_first, uint32_t count) {
	const auto& view = desc.view_info;
	// The codes that would clear this binding (DecodeColorClear), each with its color.
	std::array<uint8_t, ColorClearCheck::MaxCodes>             codes {};
	std::array<vk::ClearColorValue, ColorClearCheck::MaxCodes> colors {};
	uint32_t                                                   code_count = 0;
	for (const uint8_t code: {uint8_t {0x00}, uint8_t {0x20}, uint8_t {0x40}, uint8_t {0x80},
	                          uint8_t {0xc0}}) {
		vk::ClearColorValue color {};
		if (DecodeColorClear(desc, code, color)) {
			codes[code_count]    = code;
			colors[code_count++] = color;
		}
	}
	if (code_count == 0) {
		// The CPU check could only read the bytes and find nothing to clear.
		return true;
	}
	if (slice_size % sizeof(uint32_t) != 0 || slice_size / sizeof(uint32_t) > UINT32_MAX ||
	    view.level_count != 1) {
		return false;
	}
	{
		// The clears go through a color attachment of the backing format; anything else (block,
		// depth, aliased or unrenderable formats) keeps the CPU check.
		std::scoped_lock lock {m_lock};
		auto&            image = m_slot_images[id];
		if (image.depth_id || image.info.IsBlock() || image.info.IsDepth() ||
		    view.format != image.backing.format ||
		    !(m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
		      vk::FormatFeatureFlagBits::eColorAttachment)) {
			return false;
		}
		// A clear that does not happen must leave the guest's contents, so make them current
		// first, as ClearImage does for partial clears.
		TrackImage(id);
		if (image.IsBufferModified() || image.IsCpuDirty()) {
			InitializeImage(id);
			if (image.IsBufferModified() || image.IsCpuDirty()) {
				return false;
			}
		}
	}
	static uint64_t gpu_checks = 0;
	if (++gpu_checks % 4096 == 1) {
		LOGF("TextureCache: color metadata checked on the GPU %" PRIu64 " times\n", gpu_checks);
	}
	// Diagnostics: KYTY_GPU_CLEAR_CHECK=2 does not expand the keys; =3 records the check only (no
	// clears, no expansion), to tell its decisions from the drain it replaces.
	static const char mode = [] {
		const char* value = std::getenv("KYTY_GPU_CLEAR_CHECK");
		return value != nullptr ? value[0] : '1';
	}();
	const bool no_expand = mode == '2' || mode == '3';
	const bool expand  = desc.type != BindingType::VideoOut && !no_expand;
	auto&      command = m_scheduler.Current();
	command.EndRendering();
	// The slices' GPU copy (CPU-written bytes are uploaded first); expanding keys writes it.
	const auto [buffer, offset] = m_buffer_cache.ObtainBuffer(
	    address + slice_size * first, slice_size * count, expand, false, {}, true);
	const auto base     = buffer->BufferDeviceAddress() + offset;
	const auto recorder = command.Recorder();
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto result =
		    m_clear_check->Record(recorder, base + slice_size * slice,
		                          static_cast<uint32_t>(slice_size / sizeof(uint32_t)), codes,
		                          code_count, expand);
		std::scoped_lock lock {m_lock};
		auto&            image = m_slot_images[id];
		ImageViewInfo    layer {};
		layer.format      = view.format;
		layer.type        = vk::ImageViewType::e2D;
		layer.base_level  = view.base_level;
		layer.base_layer  = image_first + slice;
		layer.layer_count = 1;
		layer.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, recorder);
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(layer);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		const vk::Extent2D extent {std::max(image.info.extent.width >> view.base_level, 1u),
		                           std::max(image.info.extent.height >> view.base_level, 1u)};
		vk::RenderingInfo  rendering {};
		rendering.renderArea.extent    = extent;
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		for (uint32_t k = 0; k < (mode == '3' ? 0u : code_count); k++) {
			// Only vkCmdClearAttachments obeys conditional rendering; transfer clears do not.
			recorder.beginConditionalRenderingEXT(result.predicates, result.code_offset[k]);
			recorder.beginRendering(rendering);
			vk::ClearAttachment clear {};
			clear.aspectMask        = vk::ImageAspectFlagBits::eColor;
			clear.colorAttachment   = 0;
			clear.clearValue.color  = colors[k];
			const vk::ClearRect rect {vk::Rect2D {{0, 0}, extent}, 0, 1};
			recorder.clearAttachments(clear, rect);
			recorder.endRendering();
			recorder.endConditionalRenderingEXT();
		}
		CommitGpuWrite(image);
	}
	// Diagnostics: KYTY_GPU_CLEAR_CHECK_SYNC=1 submits afterwards, =2 also waits for the GPU, as
	// the CPU check's download did (to tell its wait from its bytes).
	static const int sync = [] {
		const char* value = std::getenv("KYTY_GPU_CLEAR_CHECK_SYNC");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	if (sync == 1) {
		m_scheduler.Flush();
	} else if (sync == 2) {
		m_scheduler.FlushAndWait();
	}
	return true;
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		const auto hash = image.HashGuestEdges();
		if (image.NeedsMaybeCpuHash()) {
			image.SetMaybeCpuHash(hash);
			return;
		}
		(void)image.ResolveMaybeCpuHash(hash);
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	record.depth_id             = depth_id;
	record.stencil_subresources = depth.stencil_subresources;
	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		// The page-table search below, per page every image on it, was ~10 % of the GPU thread in
		// UFC 5 fights (host sampler) for lookups that find the same image each frame. In-run A/B,
		// 2026-10-16, standing fight: +4.2 % fps (+-0.7), 9.8 against 10.7 us a draw. Needs labels
		// written at completion (LabelsAtCompletion): with early labels, fighters' meshes tore.
		static const Common::AbSwitch<bool> memo_switch("KYTY_FIND_IMAGE_MEMO", [](const char* value) {
			return value == nullptr || value[0] != '0';
		});
		const bool memo_enabled = memo_switch.Get();
		if (memo_enabled && m_find_image_memo.empty()) {
			m_find_image_memo.resize(FindImageMemoEntries);
		}
		const auto memo_index =
		    (desc.info.data.address * 0x9e3779b97f4a7c15ull ^ desc.info.data.size ^
		     static_cast<uint64_t>(desc.info.pixel_format) << 7u ^ static_cast<uint64_t>(desc.info.type) << 3u) >>
		    52u;
		FindImageMemo* memo = memo_enabled ? &m_find_image_memo[memo_index % FindImageMemoEntries] : nullptr;
		if (memo != nullptr && memo->epoch == m_image_set_epoch && memo->address == desc.info.data.address &&
		    memo->size == desc.info.data.size && memo->format == desc.info.pixel_format &&
		    memo->type == static_cast<uint32_t>(desc.info.type) && memo->exact == exact_format) {
			if (const auto* image = m_slot_images.try_get(memo->id);
			    image != nullptr && image->registered && SameBacking(image->info, desc.info, exact_format)) {
				result = memo->id;
			}
		}
		const bool memo_hit = static_cast<bool>(result);
		const auto candidates =
		    memo_hit ? ImageIds {} : FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			if (SameBacking(image.info, desc.info, exact_format)) {
				result = id;
			}
		}
		if (memo != nullptr && !memo_hit && result) {
			*memo = {m_image_set_epoch, desc.info.data.address, desc.info.data.size, result,
			         desc.info.pixel_format, static_cast<uint32_t>(desc.info.type), exact_format};
		}

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result);
				result = {};
			}
		}
		if (!result) {
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	// Only an image that starts at the address qualifies, and it covers that byte: one page to
	// query instead of every page of the range (a multi-megabyte texel buffer on every draw).
	for (const auto id: FindImagesInRegion(address, 1, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr || (ensure_valid && !owner->SafeToDownload())) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) {
			selected = owner->depth_id;
		}
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
		if (desc.type == BindingType::Texture) {
			SettleTexelAlias(id);
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.depth_target = true;
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		m_surface_metas.emplace(desc.info.metadata.range.address,
		                        MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                      .clear_mask = image.info.htile_clear_mask});
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil));
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	// Image writes bypass BufferCache::MarkGpuWritten, so a fill recorded over memory this image
	// now reuses would otherwise still answer color-metadata clear checks with stale bytes.
	if (!image.info.data.Empty()) {
		m_buffer_cache.ForgetKnownFills(image.info.data.address, image.info.data.size,
		                                image.known_fills_generation);
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	// Aliases aged by frames stay alive side by side. A single owner among them: this write
	// supersedes the bytes of every other live image overlapping it, so none of those may later
	// be written back (collector, linear readback, buffer sync) over it. Their native contents
	// stay as they are; the tick rule freed them without a write-back. Scanned only when the
	// owner changes, not on every bind.
	if (AliasAgeByFrames() && !image.alias_owner && !image.depth_id && image.registered &&
	    !image.info.data.Empty()) {
		for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			auto* other = m_slot_images.try_get(id);
			if (other == nullptr || other == &image || other->depth_id || !other->registered) {
				continue;
			}
			other->alias_owner = false;
			if (other->IsGpuModified()) {
				other->ClearGpuModified();
			}
		}
		image.alias_owner = true;
	}
	image.MarkGpuModified();
	image.gpu_write_seq = ++m_gpu_write_seq;
}

// PPSA03541
// FSR2 reads its exposure from a 1x1 texture at the base of a transient heap that the game
// reuses for its upscaled targets during the frame. On the PS5 those targets overwrite the
// exposure bytes. Kyty keeps each alias in its own native image. Copying elements between them
// follows cache eviction rather than the game's write order and fed FSR2 non-finite exposures,
// so an overwritten texel reads as cleared instead.
bool TextureCache::IsTexelAlias(const Image& image) {
	const auto& info = image.info;
	return HasGuestTexelLayout(image) && info.extent == vk::Extent3D {1, 1, 1} &&
	       info.resources.levels == 1 && info.resources.layers == 1;
}

bool TextureCache::OverwritesTexel(const Image& writer, const Image& texel) {
	return &writer != &texel && writer.registered && !writer.depth_id &&
	       writer.backing.image != nullptr && writer.info.data.address == texel.info.data.address &&
	       writer.gpu_write_seq > texel.gpu_write_seq;
}

void TextureCache::SettleTexelAlias(ImageId id) {
	KYTY_PROFILER_FINE_FUNCTION();
	auto& texel = m_slot_images[id];
	if (!IsTexelAlias(texel) || texel.binding.shader_write || texel.binding.is_target ||
	    !m_scheduler.Active() || m_scheduler.Current().IsInvalid()) {
		return;
	}
	for (const auto candidate:
	     FindImagesInRegion(texel.info.data.address, texel.info.bytes_per_block, false)) {
		const auto* writer = m_slot_images.try_get(candidate);
		// A target of the pending draw has not been written yet.
		if (writer != nullptr && !writer->binding.shader_write && !writer->binding.is_target &&
		    OverwritesTexel(*writer, texel)) {
			ClearTexelAlias(id, writer, "overwritten");
			return;
		}
	}
	if (texel.IsGpuModified()) {
		return;
	}
	// Guest bytes behind a dropped alias are stale; a non-finite float there was never a value.
	const auto format = texel.info.pixel_format;
	const auto bytes  = texel.info.bytes_per_block;
	if ((format != vk::Format::eR32Sfloat && format != vk::Format::eR32G32Sfloat &&
	     format != vk::Format::eR32G32B32A32Sfloat &&
	     format != vk::Format::eB10G11R11UfloatPack32) ||
	    bytes > 16 || bytes % 4 != 0) {
		return;
	}
	std::array<float, 4> guest {};
	if (!LibKernel::Memory::TryReadBacking(texel.info.data.address, guest.data(), bytes) ||
	    std::all_of(guest.begin(), guest.begin() + bytes / 4,
	                [](float value) { return std::isfinite(value); })) {
		return;
	}
	ClearTexelAlias(id, nullptr, "non-finite guest");
}

void TextureCache::ClearOverwrittenTexels(ImageId writer_id) {
	const auto& writer = m_slot_images[writer_id];
	if (writer.info.data.Empty() || !m_scheduler.Active() || m_scheduler.Current().IsInvalid()) {
		return;
	}
	// The freed image takes the record of its writes with it.
	for (const auto candidate: FindImagesInRegion(writer.info.data.address, 4, false)) {
		const auto* texel = m_slot_images.try_get(candidate);
		if (texel != nullptr && IsTexelAlias(*texel) && OverwritesTexel(writer, *texel)) {
			ClearTexelAlias(candidate, &writer, "freed alias");
		}
	}
}

void TextureCache::ClearTexelAlias(ImageId id, const Image* writer, std::string_view reason) {
	auto& texel = m_slot_images[id];
	// UFC 5's alias is R11G11B10 (format 122) but FSR2 reads the same 32 bits as one R32 float:
	// blue 0x3dc becomes 0x3dc00000, an exposure of 0.09375. Zero makes FSR2 use 1.0, far too
	// bright for the game's x783 working space (speckles and ghosting in the ring). Hardcoded:
	// what the PS5 really reads there is still unknown.
	ClearImage(m_scheduler.Current(), id, texel.backing.format,
	           {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
	           vk::ClearValue {vk::ClearColorValue {std::array<float, 4> {0, 0, 0.00671875, 0}}});
	const auto source = writer == nullptr
	                        ? std::string {}
	                        : fmt::format(" by {}x{} format={}", writer->info.extent.width,
	                                      writer->info.extent.height,
	                                      static_cast<uint32_t>(writer->backing.format));
	auto       kind   = fmt::format("{:#x} format={} {}{}", texel.info.data.address,
	                                static_cast<uint32_t>(texel.backing.format), reason, source);
	if (++m_texel_logs % 600 == 1 ||
	    (m_texel_log_seen.size() < 400 && m_texel_log_seen.insert(kind).second)) {
		LOGF("TextureCache: [%u] cleared texel alias %s tick=%" PRIu64 "\n", m_texel_logs,
		     kind.c_str(), m_scheduler.CurrentTick());
	}
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		ImageViewInfo view {};
		view.format = format;
		view.type   = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, command.Recorder());
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eClear;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		attachment.clearValue  = clear;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = {
		    std::max(image.info.extent.width >> range.baseMipLevel, 1u),
		    std::max(image.info.extent.height >> range.baseMipLevel, 1u)};
		rendering.layerCount           = range.layerCount;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		command.Recorder().beginRendering(&rendering);
		command.Recorder().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Recorder());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Recorder().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Recorder().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	// Every game thread's write fault comes here. Most land where no image is, and m_lock is a
	// spin lock the GPU thread takes for each texture lookup.
	if (!MayHaveImages(address, size)) {
		return;
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

bool TextureCache::MayHaveImages(uint64_t address, uint64_t size) const noexcept {
	const auto begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(begin, end - begin, pages)) {
		return true;
	}
	for (auto page = pages.first; page < pages.last_exclusive; page++) {
		if ((m_image_pages[page / 64].load(std::memory_order_seq_cst) &
		     (uint64_t {1} << (page % 64))) != 0) {
			return true;
		}
	}
	return false;
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.data.size / layers;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	auto copies = BuildDepthCopies(info, full_slice_size, vk::ImageAspectFlagBits::eDepth);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			for (auto& copy: copies) {
				copy.bufferOffset += destination_offset;
			}
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
			return;
		}
		const auto tiles = BuildDepthTiles(info);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, info.data.size,
		                  info.data.size, tiles);
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	for (uint32_t layer = 0; layer < layers; layer++) {
		copies[layer].bufferOffset = transfer_slice * layer;
	}
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto       guest_linear = tiled ? m_tiler.GetScratchBuffer(info.data.size, host_linear.buffer)
	                                : TileManager::Result {destination.Handle(), destination_offset,
	                                                       destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (!tiled) {
		return;
	}
	const auto tiles = BuildDepthTiles(info);
	m_tiler.Tile(guest_linear.buffer, guest_linear.offset, info.data.size, destination.Handle(),
	             destination_offset, info.data.size, tiles);
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                     uint64_t destination_size, ImageDownload transfer) {
	if (!transfer.valid) {
		EXIT("TextureCache: invalid image download transfer\n");
	}
	if (transfer.depth_target) {
		if (destination_size != image.info.data.size) {
			EXIT("TextureCache: partial depth image download is unsupported\n");
		}
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	auto&      texture   = transfer.texture;
	const auto transform = texture.color_transform;
	if (texture.tiles.empty()) {
		if (transform != ColorTransform::None) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.TransformColor(linear,
			                       {destination.Handle(), destination_offset, destination_size},
			                       transform, false);
			return;
		}
		for (auto& copy: texture.regions) {
			copy.bufferOffset += destination_offset;
		}
		image.Download(texture.regions, destination.Handle(), destination_offset, destination_size);
		return;
	}

	m_tiler.TileImage(image, texture.regions, destination.Handle(), destination_offset,
	                  destination_size, texture.LinearSize(), texture.tiles, transform);
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (image.info.IsVolume()) {
		// Volume mips contain strided block slices, so a mip's linear span cannot prove that
		// every retained slice fits. Keep volume synchronization whole-image only.
		if (!buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
			return false;
		}
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	auto transfer = m_texture_cache.BuildDownload(image);
	if (!transfer.valid) {
		return false;
	}
	if (transfer.depth_target && copy_size != image.info.data.size) {
		return false;
	}
	if (!transfer.depth_target && levels < image.info.resources.levels) {
		auto& texture = transfer.texture;
		std::erase_if(texture.regions, [levels](const vk::BufferImageCopy& region) {
			return region.imageSubresource.mipLevel >= levels;
		});
		if (texture.regions.empty()) {
			return false;
		}
		if (!texture.tiles.empty()) {
			texture.tiles.clear();
			if (!TextureBuildGpuTileInfos(copy_size, texture.regions, texture.layout, levels,
			                              texture.tiles)) {
				return false;
			}
		}
	}
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, std::move(transfer));
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id, bool evicting,
                                       std::shared_ptr<std::atomic<bool>>* landed) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid || !image.SafeToDownload()) {
		return false;
	}
	const auto range = image.info.data;
	auto&      ring  = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	// An image larger than the download ring gets a staging buffer of its own for this one
	// transfer: the ring cannot map it, and its regions cannot be cut into ring-sized pieces
	// the way a buffer download can.
	std::shared_ptr<Buffer> dedicated;
	Buffer*                 download = &ring;
	uint8_t*                mapped   = nullptr;
	uint64_t                offset   = 0;
	if (range.size > ring.Size()) {
		dedicated = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     AllFlags, range.size);
		download  = dedicated.get();
		mapped    = dedicated->Mapped().data();
		if (mapped == nullptr) {
			return false;
		}
	} else {
		std::tie(mapped, offset) =
		    ring.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
		if (mapped == nullptr) {
			EXIT("TextureCache: failed to map reusable download buffer\n");
		}
		ring.Commit();
	}
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	const bool guard      = evicting && EvictWriteBackGuard();
	uint64_t   guest_hash = 0;
	if (guard && !HashGuestBacking(range.address, range.size, guest_hash)) {
		return false;
	}
	download->Flush(offset, range.size);

	DownloadImage(image, *download, offset, range.size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	std::shared_ptr<std::atomic<bool>> landed_flag;
	if (landed != nullptr) {
		landed_flag = std::make_shared<std::atomic<bool>>(false);
		*landed     = landed_flag;
	}
	{
		std::lock_guard lock(m_writeback_mutex);
		m_writebacks_in_flight.emplace_back(range.address, range.size);
	}
	m_scheduler.DeferPriorityOperation([this, download, dedicated, range, mapped, offset, guard,
	                                    guest_hash, landed_flag] {
		struct Landed {
			TextureCache&                             cache;
			const GuestRange&                         range;
			const std::shared_ptr<std::atomic<bool>>& flag;
			~Landed() {
				{
					std::lock_guard lock(cache.m_writeback_mutex);
					auto&           list = cache.m_writebacks_in_flight;
					const auto      entry =
					    std::find(list.begin(), list.end(), std::pair {range.address, range.size});
					if (entry != list.end()) {
						*entry = list.back();
						list.pop_back();
					}
				}
				if (flag != nullptr) flag->store(true, std::memory_order_release);
			}
		} landed_scope {*this, range, landed_flag};
		download->Invalidate(offset, range.size);
		uint64_t current = 0;
		if (guard && (!HashGuestBacking(range.address, range.size, current) ||
		              current != guest_hash)) {
			// The guest wrote (or unmapped) these bytes after the image was evicted: they are
			// newer than its pixels.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("TextureCache: evicted image write-back dropped, guest=0x%016" PRIx64
				     " size=0x%" PRIx64 ": the guest bytes changed before it landed\n",
				     range.address, range.size);
			}
			return;
		}
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
	});
	return true;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->Overlaps(address, size)) {
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner->info.data.address;
		const auto image_end   = owner->info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	return (found->second.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

bool TextureCache::IsWriteBackInFlight(uint64_t address, uint64_t size) {
	std::lock_guard lock(m_writeback_mutex);
	return std::any_of(
	    m_writebacks_in_flight.begin(), m_writebacks_in_flight.end(), [&](const auto& entry) {
		    return entry.first < address + size && address < entry.first + entry.second;
	    });
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

uint64_t TextureCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

namespace {
// Why the collector does not free: over each TexStats window, candidates it visited and what
// happened to them.
struct GcSkipStats {
	uint64_t visited      = 0;
	uint64_t freed        = 0;
	uint64_t freed_mb     = 0;
	uint64_t gone         = 0; // no longer registered
	uint64_t depth        = 0; // part of a depth/stencil association
	uint64_t gpu_tiled    = 0; // GPU-written tiled: needs a write-back it cannot do
	uint64_t gpu_unpress  = 0; // GPU-written, not under pressure
	uint64_t gpu_download = 0; // GPU-written, write-back failed
	uint64_t young        = 0; // runs where the LRU had nothing old enough
	uint64_t written_back = 0; // GPU-written tiled images written back, then freed
	uint64_t bindless     = 0; // bindless images freed: unsampled in the last usage probe
	uint64_t small_target = 0; // render targets of at most 1024x1024 kept (glyphs, cached layers)
	uint64_t pending      = 0; // evicted, write-back not landed yet
	uint64_t revived      = 0; // used again before the write-back landed: stays
};
GcSkipStats g_gc_skip;
} // namespace

void TextureCache::ReleaseDroppedByGuest(ImageId id) {
	std::scoped_lock lock {m_lock};
	if (auto* image = m_slot_images.try_get(id); image != nullptr) {
		image->frame_guest_dropped = m_graphics.presented_frames.load(std::memory_order_relaxed);
		m_guest_dropped.push_back(id);
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	m_gc_tick++;
	const uint64_t clock = LruClock();
	// Evictions whose write-back has landed: free the image now, unless it was used since (then
	// its pixels are newer than the guest bytes and it stays, as if never a candidate).
	std::erase_if(m_pending_evictions, [&](const PendingEviction& pending) {
		if (!pending.landed->load(std::memory_order_acquire)) {
			return false;
		}
		m_evicting.erase(pending.id);
		auto* image = m_slot_images.try_get(pending.id);
		if (image == nullptr || !image->registered || image->info.data.address != pending.address) {
			return true;
		}
		if (image->frame_accessed_last != pending.frame) {
			g_gc_skip.revived++;
			return true;
		}
		const auto freed = image->AccountedSize();
		g_gc_skip.freed++;
		g_gc_skip.freed_mb += freed >> 20u;
		FreeImage(pending.id);
		return true;
	});
	// Memory pressure (GraphicContext::MemoryShortfall: device use past 90 % of the video memory
	// target): the thresholds sit that far below what the cache holds, so the collector frees its
	// oldest images until there is room again.
	if (m_trigger_gc_memory != m_applied_trigger_gc_memory ||
	    m_pressure_gc_memory != m_applied_pressure_gc_memory ||
	    m_critical_gc_memory != m_applied_critical_gc_memory) {
		m_base_trigger_gc_memory  = m_trigger_gc_memory;
		m_base_pressure_gc_memory = m_pressure_gc_memory;
		m_base_critical_gc_memory = m_critical_gc_memory;
	}
	if (const auto shortfall = std::max(m_graphics.MemoryShortfall(), m_graphics.ForcedShortfall(false));
	    shortfall != 0) {
		const auto collectable = CollectableMemory();
		const auto cap =
		    std::max<uint64_t>(collectable > shortfall ? collectable - shortfall : 0, 512ull << 20u);
		m_trigger_gc_memory  = std::min(m_base_trigger_gc_memory, cap);
		m_pressure_gc_memory = std::min(m_base_pressure_gc_memory, cap);
		m_critical_gc_memory = std::min(m_base_critical_gc_memory, cap + cap / 8);
	} else {
		m_trigger_gc_memory  = m_base_trigger_gc_memory;
		m_pressure_gc_memory = m_base_pressure_gc_memory;
		m_critical_gc_memory = m_base_critical_gc_memory;
	}
	m_applied_trigger_gc_memory  = m_trigger_gc_memory;
	m_applied_pressure_gc_memory = m_pressure_gc_memory;
	m_applied_critical_gc_memory = m_critical_gc_memory;
	if (!m_guest_dropped.empty()) {
		// Engine method: the guest's own streaming dropped these textures from its heaps, as an
		// engine unloads a texture; free them once no draw has used them for a few frames, not
		// when the cache reaches its memory threshold. Images the GPU wrote keep their data for
		// the regular collector (it writes them back); one a heap entry or a render target uses
		// again stays.
		const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
		std::erase_if(m_guest_dropped, [&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr || !image->registered || image->bindless_pinned ||
			    image->binding.is_target || image->IsGpuModified()) {
				return true;
			}
			// A grace of 300 frames (10-20 s): the guest drops entries and takes the same texture
			// back shortly after (the jungle slash stalled 13 s with a grace of 3 frames, each
			// texture freed and created again). The drop time counts as a use.
			if (std::max(image->frame_accessed_last, image->frame_guest_dropped) + 300 > frame) {
				return false; // used or dropped lately: look again later
			}
			m_guest_dropped_freed++;
			m_guest_dropped_freed_bytes += image->AccountedSize();
			FreeImage(id);
			return true;
		});
		if (m_gc_tick % 500 == 0) {
			LOGF("TexGuestDrop: frame=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) waiting=%zu\n",
			     frame, m_guest_dropped_freed, m_guest_dropped_freed_bytes >> 20u,
			     m_guest_dropped.size());
		}
	}
	if (m_graphics.CanReportMemoryUsage()) {
		// Pressure is judged by this cache's own image bytes, which RegisterImage and
		// UnregisterImage already track. Device-wide usage counts the images this cache has
		// spilled to host memory (and every buffer), so it reads "critical" no matter how
		// small the cache is -- and never falls, so the collector cannot tell that it is the
		// one holding too much. Spider-Man's world then grew to 7.6 GB of images on a 6 GB
		// card, all of it spilled, and every shader access crossed PCIe (3 fps).
		// KYTY_GC_DEVICE_BYTES=1 restores the old policy, as in the buffer cache.
		static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
		if (device_bytes) {
			m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
		}
	}
	if (m_pinned_frame != clock) {
		// Heap textures follow the guest (settled when it writes an entry, freed when it drops
		// them) and the collector cannot free them; counted against the bound they kept the cache
		// over it, and it freed every other image again and again (33,916 in 4 minutes of a new
		// game, the frame rate a fifth lower). The bounds apply to the images it can free.
		m_pinned_frame = clock;
		m_pinned_bytes = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered && image.bindless_pinned) {
				m_pinned_bytes += image.AccountedSize();
			}
		});
	}
	if (CollectableMemory() < m_trigger_gc_memory) {
		return;
	}
	// The collector runs once per guest submission -- this title submits hundreds of times a
	// frame -- so a budget counted per run is really that many evictions per frame, and every
	// eviction is an image recreated, synchronized and submitted. Budget the work per frame
	// instead: 7 600 submissions a frame came from a per-run budget of 256.
	if (m_gc_budget_frame != clock) {
		m_gc_budget_frame       = clock;
		m_gc_freed_bytes_frame  = 0;
		m_gc_freed_images_frame = 0;
		m_gc_written_back_bytes_frame = 0;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool pressured  = CollectableMemory() >= m_pressure_gc_memory;
		bool aggressive = allow_aggressive && CollectableMemory() >= m_critical_gc_memory;
		// Ages in frames. This collector runs once per guest submission, and a title can
		// submit a hundred times a frame: an age counted in its ticks evicted textures inside
		// the frame that used them, to recreate them the next (11 000 images a second at the
		// title menu). An image used in the frame being recorded is never a candidate, at any
		// pressure; over budget the collector reaches back only as far as the previous frame.
		const uint64_t age = std::min<uint64_t>(aggressive ? 1 : pressured ? 4 : 16, clock);
		// Under pressure the work is budgeted per frame, in bytes actually freed, and in
		// proportion to the excess, so the cache converges on its threshold instead of either
		// churning (a per-run budget: 7 600 submissions a frame) or growing without bound (a
		// small fixed count, spent on GPU-written render targets that are skipped: 13.5 GB).
		// Only frees count against it. A cap on images per frame bounds the synchronization
		// that recreating them costs.
		constexpr uint64_t MiB             = 1024 * 1024;
		constexpr size_t   MaxFreesInFrame = 1024;
		uint64_t           byte_budget     = 0;
		size_t             deletions       = 10;
		if (pressured || aggressive) {
			const auto threshold = aggressive ? m_critical_gc_memory : m_pressure_gc_memory;
			const auto excess    = CollectableMemory() - threshold;
			byte_budget = aggressive ? std::max<uint64_t>(64 * MiB, excess / 4)
			                         : std::max<uint64_t>(16 * MiB, excess / 8);
			if (m_gc_freed_bytes_frame >= byte_budget ||
			    m_gc_freed_images_frame >= MaxFreesInFrame) {
				return;
			}
			deletions = MaxFreesInFrame - m_gc_freed_images_frame;
		}
		// Skipped candidates (GPU-written tiled images, depth associations) do not use up the
		// budget, so look further down the LRU than the number of frees wanted.
		const size_t         visit_limit = pressured || aggressive ? deletions * 4 : deletions;
		std::vector<ImageId> candidates;
		candidates.reserve(std::min<size_t>(visit_limit, 4096));
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(clock - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() >= visit_limit || candidates.size() >= 4096;
		});
		if (candidates.empty() && (pressured || aggressive)) {
			g_gc_skip.young++;
		}
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			if (!(pressured || aggressive)) {
				--deletions;
			}
			g_gc_skip.visited++;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered) {
				g_gc_skip.gone++;
				continue;
			}
			if (owner->depth_id) {
				g_gc_skip.depth++;
				continue;
			}
			if (m_evicting.contains(id)) {
				g_gc_skip.pending++;
				continue;
			}
			// Small render targets are the layers the game draws once and samples from then on
			// (glyph atlases, cached UI): a few MB each, and a lost one is lost text (letters went
			// missing when they were evicted under "critical", 2026-10-09). They are never evicted;
			// the guest dropping them (ReleaseDroppedByGuest) is what frees them.
			if (owner->usage.render_target && owner->info.extent.width <= 1024 &&
			    owner->info.extent.height <= 1024) {
				g_gc_skip.small_target++;
				continue;
			}
			if (owner->bindless_pinned) {
				// Pinned for bindless draws; under pressure one no draw sampled during the last
				// usage probe may go (ISSUES #15: on 4-8 GB cards these grew to 2-5 GB and never
				// shrank).
				if (!(pressured || aggressive) || !m_bindless_evictable.contains(id)) {
					continue;
				}
				m_bindless_evictable.erase(id);
				g_gc_skip.bindless++;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				// A GPU-written tiled image was never evicted: render targets and textures the
				// GPU decompresses are almost all of a 4K title's images, so under pressure the
				// collector found nothing else it could free (Spider-Man: 3 500 of every 4 000
				// candidates) and the cache grew to 13.5 GB until the driver crashed in
				// present. Under pressure such an image is now written back, detiled, like any
				// other, within a per-frame write-back budget; candidates are never ones used in
				// the frame being recorded. KYTY_GC_KEEP_TILED=1 restores the old rule.
				static const bool keep_tiled = std::getenv("KYTY_GC_KEEP_TILED") != nullptr;
				constexpr uint64_t WriteBackPerFrame = 256 * MiB;
				if (safe && owner->info.IsTiled() &&
				    (keep_tiled || !(pressured || aggressive) ||
				     (m_gc_written_back_bytes_frame != 0 &&
				      m_gc_written_back_bytes_frame + owner->info.data.size > WriteBackPerFrame))) {
					g_gc_skip.gpu_tiled++;
					continue;
				}
				if (safe && !pressured) {
					g_gc_skip.gpu_unpress++;
					continue;
				}
				if (!safe) {
					// GPU-written, and CPU- or copy-written as well: its contents exist only in
					// this image and cannot be written back as they are. Freeing it lost them:
					// under a 6 GiB bound the Leap Attack prompt read "L p Att k" (the glyph atlas,
					// filled by both). It stays until it is safe or the guest drops it.
					g_gc_skip.gpu_download++;
					continue;
				}
				std::shared_ptr<std::atomic<bool>> landed;
				if (safe && !DownloadImageMemory(id, true, &landed)) {
					g_gc_skip.gpu_download++;
					continue;
				}
				if (safe && owner->info.IsTiled()) {
					m_gc_written_back_bytes_frame += owner->info.data.size;
					g_gc_skip.written_back++;
				}
				// Freed once the write-back has landed (the top of the next run); the bytes count
				// against this frame's budget now, since they are going.
				m_pending_evictions.push_back(
				    {id, owner->info.data.address, owner->frame_accessed_last, std::move(landed)});
				m_evicting.insert(id);
				if (pressured || aggressive) {
					--deletions;
					m_gc_freed_images_frame++;
					m_gc_freed_bytes_frame += owner->AccountedSize();
					if (m_gc_freed_bytes_frame >= byte_budget) {
						break;
					}
				}
				continue;
			}
			const auto freed = owner->AccountedSize();
			g_gc_skip.freed++;
			g_gc_skip.freed_mb += freed >> 20u;
			FreeImage(id);
			if (pressured || aggressive) {
				--deletions;
				m_gc_freed_images_frame++;
				m_gc_freed_bytes_frame += freed;
				if (m_gc_freed_bytes_frame >= byte_budget) {
					break;
				}
			}
			if (CollectableMemory() < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (CollectableMemory() < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (CollectableMemory() >= m_critical_gc_memory) {
		collect(true);
	}
	if (m_gc_tick % 500 == 0) {
		uint64_t images = 0;
		uint64_t bytes  = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered) {
				images++;
				bytes += image.AccountedSize();
			}
		});
		LOGF("TexStats: runs=%" PRIu64 " frame=%" PRIu64 " images=%" PRIu64 " image_bytes=%" PRIu64
		     "MB device_used=%" PRIu64 "MB pressure=%" PRIu64 "MB critical=%" PRIu64 "MB\n",
		     m_gc_tick, m_graphics.presented_frames.load(std::memory_order_relaxed), images,
		     bytes >> 20u, m_total_used_memory >> 20u, m_pressure_gc_memory >> 20u,
		     m_critical_gc_memory >> 20u);
		LOGF("TexGc: visited=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) written_back=%" PRIu64
		     " gone=%" PRIu64 " depth=%" PRIu64 " gpu_tiled=%" PRIu64 " gpu_unpressured=%" PRIu64
		     " gpu_download_failed=%" PRIu64 " nothing_old_enough=%" PRIu64 " bindless=%" PRIu64
		     " small_target=%" PRIu64 " pending=%" PRIu64 " revived=%" PRIu64 "\n",
		     g_gc_skip.visited, g_gc_skip.freed, g_gc_skip.freed_mb, g_gc_skip.written_back,
		     g_gc_skip.gone, g_gc_skip.depth, g_gc_skip.gpu_tiled, g_gc_skip.gpu_unpress,
		     g_gc_skip.gpu_download, g_gc_skip.young, g_gc_skip.bindless, g_gc_skip.small_target,
		     g_gc_skip.pending, g_gc_skip.revived);
		g_gc_skip = {};
	}
}

void TextureCache::SetBindlessEvictable(std::span<const ImageId> ids) {
	std::scoped_lock lock {m_lock};
	m_bindless_evictable.clear();
	m_bindless_evictable.insert(ids.begin(), ids.end());
}

void TextureCache::ProcessDownloadImages() {
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
