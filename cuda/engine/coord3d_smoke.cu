// Smoke test for coord3d.cuh — emits per-subcomponent y breakdown for
// equivalence debugging against compute_3d_octant_coords.py.
#include <stdio.h>
#include <string.h>

#include "movegen.cuh"
#include "coord3d.cuh"

using namespace engine;

int main(int /*argc*/, char** /*argv*/) {
    char line[512];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0 || line[0] == '#') continue;
        Position p;
        parse_fen(line, &p);
        // Replicate compute_coord3d step by step so we can emit each
        // subcomponent + raw side values.
        int material = material_balance_white_pov(&p);
        float y_material = 0.4f * (material / 1000.0f);

        float ks_w = king_safety_score(&p, WHITE_SIDE);
        float ks_b = king_safety_score(&p, BLACK_SIDE);
        float y_king_safety = 0.3f * (ks_w - ks_b);

        float st_w = structural_score(&p, WHITE_SIDE);
        float st_b = structural_score(&p, BLACK_SIDE);
        float y_structure = 0.2f * (st_w - st_b);

        int mob_w = mobility_count(&p, WHITE_SIDE);
        int mob_b = mobility_count(&p, BLACK_SIDE);
        int mob_total = mob_w + mob_b;
        if (mob_total < 1) mob_total = 1;
        float y_mobility = 0.1f * (float(mob_w - mob_b) / float(mob_total));

        Coord3D c = compute_coord3d(&p);
        // Format: x y z oct y_mat y_ks y_st y_mob ks_w ks_b st_w st_b mob_w mob_b
        printf("%.6f %.6f %.6f %d "
               "%.6f %.6f %.6f %.6f "
               "%.6f %.6f %.6f %.6f "
               "%d %d\n",
               c.x, c.y, c.z, c.octant_id,
               y_material, y_king_safety, y_structure, y_mobility,
               ks_w, ks_b, st_w, st_b,
               mob_w, mob_b);
    }
    return 0;
}
