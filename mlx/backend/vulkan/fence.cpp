#include "mlx/fence.h"

#include <mutex>

#include "mlx/event.h"

namespace mlx::core {

namespace {

struct FenceImpl {
  uint32_t count{0};
  std::mutex mutex;
  Stream stream;
  Event event;

  explicit FenceImpl(Stream stream) : stream(stream), event(stream) {}
};

} // namespace

Fence::Fence(Stream stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{stream}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto* impl = static_cast<FenceImpl*>(fence_.get());
  Event event;
  uint32_t target;
  Stream producer_stream{0, Device::cpu};
  {
    std::lock_guard<std::mutex> lock(impl->mutex);
    target = impl->count;
    producer_stream = impl->stream;
    event = impl->event;
  }
  event.set_value(target);
  if (producer_stream.device == Device::gpu &&
      stream.device == Device::gpu) {
    event.wait(stream);
  } else {
    event.wait();
  }
}

void Fence::update(Stream stream, const array&, bool) {
  auto* impl = static_cast<FenceImpl*>(fence_.get());
  Event event;
  uint32_t target;
  {
    std::lock_guard<std::mutex> lock(impl->mutex);
    target = ++impl->count;
    event = impl->event;
  }
  event.set_value(target);
  event.signal(stream);
}

} // namespace mlx::core
