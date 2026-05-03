"""Microbench: make_move_batched at N=8 vs serial 8 make_moves."""
import sys, time
sys.path.insert(0, '/home/frosty40/VC_V1/cuda-dojo-public/combat_shipping/live_arbiter')
import dojo_ref

p = dojo_ref.Position.startpos()
legal = p.legal_moves()
positions = [p.make_move(m) for m in legal[:8]]
ucis_per = [pos.legal_moves()[0] for pos in positions]

N = 500
t0 = time.monotonic()
for _ in range(N):
    for pos, u in zip(positions, ucis_per):
        pos.make_move(u)
dt_serial = time.monotonic() - t0

t0 = time.monotonic()
for _ in range(N):
    dojo_ref.make_move_batched(positions, ucis_per)
dt_batched = time.monotonic() - t0
print(f'serial 8x make_move: {dt_serial*1000/N:.3f} ms (per call {dt_serial*1000/N/8:.4f} ms)')
print(f'batched make_move(8): {dt_batched*1000/N:.3f} ms')
print(f'speedup of batched-8: {dt_serial/dt_batched:.2f}x')
