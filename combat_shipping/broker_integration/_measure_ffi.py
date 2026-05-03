"""Microbenchmark: per-position dojo_ref FFI vs batched FFI."""
import sys, time
sys.path.insert(0, '/home/frosty40/VC_V1/cuda-dojo-public/combat_shipping/live_arbiter')
import dojo_ref

p = dojo_ref.Position.startpos()
legal = p.legal_moves()
positions = [p.make_move(m) for m in legal[:8]]

N = 500
t0 = time.monotonic()
for _ in range(N):
    for pos in positions:
        pos.legal_moves()
dt_serial = time.monotonic() - t0
print(f'serial 8x legal_moves: {dt_serial*1000/N:.3f} ms per round of 8 ({dt_serial*1000/N/8:.4f} ms each)')

t0 = time.monotonic()
for _ in range(N):
    dojo_ref.legal_moves_batched(positions)
dt_batched = time.monotonic() - t0
print(f'batched legal_moves(8): {dt_batched*1000/N:.3f} ms per call')
print(f'speedup of batched-8 over serial-8: {dt_serial/dt_batched:.2f}x')

t0 = time.monotonic()
for _ in range(N*8):
    positions[0].legal_moves()
dt1 = time.monotonic() - t0
print(f'single legal_moves(): {dt1/(N*8)*1e6:.2f} us')

# Also is_check_batched
t0 = time.monotonic()
for _ in range(N):
    for pos in positions:
        pos.is_check()
dt_serial_chk = time.monotonic() - t0
t0 = time.monotonic()
for _ in range(N):
    dojo_ref.is_check_batched(positions)
dt_batched_chk = time.monotonic() - t0
print(f'serial 8x is_check: {dt_serial_chk*1000/N:.3f} ms')
print(f'batched is_check(8): {dt_batched_chk*1000/N:.3f} ms')
print(f'speedup is_check: {dt_serial_chk/dt_batched_chk:.2f}x')

# Larger batch: 24
positions24 = []
for m in legal[:8]:
    p1 = p.make_move(m)
    for m2 in p1.legal_moves()[:3]:
        positions24.append(p1.make_move(m2))
positions24 = positions24[:24]
print(f'len positions24 = {len(positions24)}')

t0 = time.monotonic()
for _ in range(N):
    for pos in positions24:
        pos.legal_moves()
dt_serial24 = time.monotonic() - t0
t0 = time.monotonic()
for _ in range(N):
    dojo_ref.legal_moves_batched(positions24)
dt_batched24 = time.monotonic() - t0
print(f'serial 24x legal_moves: {dt_serial24*1000/N:.3f} ms')
print(f'batched legal_moves(24): {dt_batched24*1000/N:.3f} ms')
print(f'speedup at N=24: {dt_serial24/dt_batched24:.2f}x')
