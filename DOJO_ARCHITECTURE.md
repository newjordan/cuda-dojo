# The Dojo System — FrostD4D Training Architecture

## Philosophy
v3 razoring is #2 on the ladder by beating noobs. That's not chess mastery.
v5 queenguard drew Lozza 5 times at 2155 Elo. That's survival instinct.
We build the dojos to turn survival instinct into mastery.

## The Dojos

### 🥋 DOJO 1: The Opening Temple
**Sensei**: Opening book (The UN, 200KB)
**Training**: For every known position, give the student the GM-approved move instantly.
No thinking wasted on solved positions. Save all compute for the unknown.
**Graduation**: Student plays first 15 moves at GM level without thinking.

### 🥋 DOJO 2: The Tactical Crucible
**Sensei**: Lozza (2155 Elo) via coached ralph
**Training**: Play position → Lozza disagrees → analyze WHY → adjust weights.
Focus: forks, pins, hanging pieces, back rank, sacrifices.
**Graduation**: Student agrees with Lozza on 80%+ of tactical positions.

### 🥋 DOJO 3: The Positional Forge
**Sensei**: Lozza + CUDA batch evaluation
**Training**: Thousands of quiet middlegame positions.
For each: FrostD4D eval vs Lozza eval. Minimize disagreement.
Train the mini brain weights until our eval matches Lozza's judgment.
**Graduation**: Eval correlation with Lozza > 0.85

### 🥋 DOJO 4: The Endgame Academy
**Sensei**: Endgame tables + Lozza
**Training**: KPK, KRK, KRKP, pawn endgames, rook endgames.
Perfect play from tablebases. No approximation — exact solutions.
**Graduation**: Student plays endgame positions perfectly.

### 🥋 DOJO 5: The Sparring Ring
**Sensei**: Deeper Blue (1460) + Lozza (2155) + Trainers
**Training**: Full games with post-game Lozza review.
The Cavalry activates when opportunity detected.
Spy network feeds intel. Controller routes to correct brain.
**Graduation**: Beat Deeper Blue 50%+, draw Lozza 50%+.

### 🥋 DOJO 6: The War Room (CUDA)
**Sensei**: GPU MCTS + batch eval
**Training**: 10,000 parallel position evaluations per second.
Monte Carlo tree search provides strategic second opinion.
Hybrid alpha-beta + MCTS voting system.
**Graduation**: GPU and CPU agree on best move in critical positions.

## Data Flow

```
┌──────────────────────────────────────────────────────┐
│                    THE DOJOS                          │
│                                                       │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Opening  │ │Tactical  │ │Position  │ │Endgame   │ │
│  │Temple   │ │Crucible  │ │Forge     │ │Academy   │ │
│  │(book)   │ │(Lozza)   │ │(CUDA)    │ │(tables)  │ │
│  └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │           │            │             │        │
│       └───────────┴─────┬──────┴─────────────┘        │
│                         │                             │
│                    ┌────▼────┐                        │
│                    │SPARRING │                        │
│                    │RING     │                        │
│                    └────┬────┘                        │
│                         │                             │
│                    ┌────▼────┐                        │
│                    │WAR ROOM │                        │
│                    │(CUDA)   │                        │
│                    └────┬────┘                        │
└─────────────────────────┼────────────────────────────┘
                          │
                    ┌─────▼──────┐
                    │ FROST D4D  │
                    │ THE MASTER │
                    └────────────┘
```

## Pipeline per Training Cycle

1. Student enters Opening Temple → plays book moves
2. Out of book → Tactical Crucible activates → coached ralph
3. Middlegame → Positional Forge → weight calibration from Lozza evals
4. Endgame → Endgame Academy → exact play from tables
5. Post-game → Sparring Ring → full review, weight updates
6. Batch processing → War Room → GPU-accelerated eval refinement

## Calibration Output
Each dojo produces a DELTA VECTOR — specific weight adjustments.
All vectors are combined with confidence weights.
The combined vector is applied to produce the next generation.

## Infrastructure
- `dojo_master.js` — Orchestrates all dojos, manages training cycles
- `dojo_opening.js` — Opening Temple (imports The UN book)
- `dojo_tactical.js` — Tactical Crucible (coached ralph + Lozza)
- `dojo_positional.js` — Positional Forge (CUDA batch eval)
- `dojo_endgame.js` — Endgame Academy (tablebase lookup)
- `dojo_sparring.js` — Sparring Ring (full games + review)
- `dojo_warroom.js` — War Room (CUDA MCTS + hybrid)

## Current Status
- [x] Opening Temple: opening_books.js built (1205 positions)
- [x] Tactical Crucible: coached_ralph.js built
- [x] Sparring Ring: nuclear_forge.js + combat_loop.js
- [ ] Positional Forge: needs CUDA bridge
- [ ] Endgame Academy: needs tablebases
- [x] War Room: CUDA MCTS subagent building
- [ ] Dojo Master: needs orchestrator

## Target
FrostD4D graduates all 6 dojos.
Plays GM-level openings, Lozza-grade tactics, perfect endgames.
The Cavalry charges when opportunity arises.
The Spy Network watches everything.
The UN briefs every decision.

#2 on the ladder by beating noobs is yesterday.
Tomorrow we beat the masters.
