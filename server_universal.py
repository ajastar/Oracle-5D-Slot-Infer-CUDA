import struct
import mmap
import os
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes (Localhost Client Support)

@app.route('/')
def index():
    try:
        return send_file('juggler_oracle_client.html')
    except Exception as e:
        return str(e), 404

# --- Configuration for Level 9 (Full Spectrum) ---
# Bin Layout: 
#   c1 (u32), c2 (u32), c3 (u32), c4 (u32), c5 (u32), c6 (u32)
# Total Size: 24 bytes
STRUCT_SIZE = 24
STRUCT_FMT = 'IIIIII'  # Unsigned Int x 6

# File Paths
DATA_DIR = "./" 
FILES = {
    "ultra": "matrix_db_5d_ultra_cuda.bin",
    "my": "matrix_db_5d_my_cuda.bin",
    "happy": "matrix_db_5d_happy_cuda.bin",
    "gogo": "matrix_db_5d_gogo_cuda.bin",
    "funky": "matrix_db_5d_funky_cuda.bin",
    "mr": "matrix_db_5d_mr_cuda.bin",
    "im": "matrix_db_5d_im_cuda.bin" 
}

mmaps = {}

def load_maps():
    for key, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                f = open(path, "rb")
                mmaps[key] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                print(f"Loaded {key}: {filename}")
            except Exception as e:
                print(f"Failed to load {key}: {e}")

# --- Grid Logic (Matches cuda_matrix.h V6) ---
K1_MIN, K1_STEP = 90.0, 2.0
K1_BINS = int((200.0 - 90.0) / 2.0) + 1 
K2_MIN, K2_STEP = 180.0, 10.0
K2_BINS = int((800.0 - 180.0) / 10.0) + 1
K3_MIN, K3_STEP = 90.0, 2.0
K3_BINS = int((200.0 - 90.0) / 2.0) + 1
K4_MIN, K4_STEP = 0, 250
K4_MAX = 9000
K4_BINS = (K4_MAX - K4_MIN) // K4_STEP + 1
K5_MIN, K5_STEP = 0, 250 # Finer logic (User Request)
K5_MAX = 9000
K5_BINS = (K5_MAX - K5_MIN) // K5_STEP + 1

STRIDE_K5 = 1
STRIDE_K4 = STRIDE_K5 * K5_BINS
STRIDE_K3 = STRIDE_K4 * K4_BINS
STRIDE_K1 = STRIDE_K3 * K3_BINS
STRIDE_K2 = STRIDE_K1 * K1_BINS
HEADER_SIZE = 64 

# Helper for Radius Search
def scan_area(mm, base_indices, radius):
    file_size = mm.size()
    
    total_acc = 0
    c_acc = [0] * 7 # 1-6 used
    bins_found = 0
    
    base_k1, base_k2, base_k3, base_k4, base_k5 = base_indices

    # Optimization: Pre-calc bounds to avoid 5-nested loop overhead if possible
    # But for 3^5 or 5^5, simple loops are acceptable in Python given mmap speed.
    
    for d1 in range(-radius, radius+1):
        idx_k1 = base_k1 + d1
        if idx_k1 < 0 or idx_k1 >= K1_BINS: continue
        
        for d2 in range(-radius, radius+1):
            idx_k2 = base_k2 + d2
            if idx_k2 < 0 or idx_k2 >= K2_BINS: continue
            
            for d3 in range(-radius, radius+1):
                idx_k3 = base_k3 + d3
                if idx_k3 < 0 or idx_k3 >= K3_BINS: continue
                
                for d4 in range(-radius, radius+1):
                    idx_k4 = base_k4 + d4
                    if idx_k4 < 0 or idx_k4 >= K4_BINS: continue
                    
                    for d5 in range(-radius, radius+1):
                        idx_k5 = base_k5 + d5
                        if idx_k5 < 0 or idx_k5 >= K5_BINS: continue

                        flat_idx = (idx_k2 * STRIDE_K2 + 
                                    idx_k1 * STRIDE_K1 + 
                                    idx_k3 * STRIDE_K3 + 
                                    idx_k4 * STRIDE_K4 + 
                                    idx_k5 * STRIDE_K5)
                        
                        offset = HEADER_SIZE + flat_idx * STRUCT_SIZE
                        
                        if offset + STRUCT_SIZE <= file_size:
                            mm.seek(offset)
                            raw = mm.read(STRUCT_SIZE)
                            # Unpack
                            vals = struct.unpack(STRUCT_FMT, raw)
                            sub_total = sum(vals)
                            
                            if sub_total > 0:
                                total_acc += sub_total
                                for i in range(6):
                                    c_acc[i+1] += vals[i]
                                bins_found += 1
                                
    return total_acc, c_acc, bins_found

@app.route('/query', methods=['POST'])
def query_bin():
    data = request.json
    machine = data.get('machine', 'im')
    
    if machine not in mmaps:
        return jsonify({"error": "Machine DB not loaded"}), 404

    try:
        # Input Parsing
        k1 = float(data['k1'])
        k2 = float(data['k2'])
        k3 = float(data['k3'])
        k4 = int(data['k4'])
        k5 = int(data['k5'])
        
        # Base Indices
        b_k1 = int((k1 - K1_MIN) / K1_STEP)
        b_k2 = int((k2 - K2_MIN) / K2_STEP)
        b_k3 = int((k3 - K3_MIN) / K3_STEP)
        b_k4 = int(k4 / K4_STEP)
        b_k5 = int(k5 / K5_STEP)
        
        base_indices = (b_k1, b_k2, b_k3, b_k4, b_k5)
        mm = mmaps[machine]
        
        # Adaptive Search Strategy
        # Level 1: Precise (Radius 1) - 3^5 = 243 checks
        # Level 2: Broad (Radius 2) - 5^5 = 3125 checks
        # Level 3: Wide (Radius 3) - 7^5 = 16807 checks
        
        final_total = 0
        final_c = []
        final_bins = 0
        used_radius = 0
        
        for r in [1, 2, 3]:
            t, c, b = scan_area(mm, base_indices, r)
            if t > 0:
                final_total = t
                final_c = c
                final_bins = b
                used_radius = r
                break # Stop as soon as we find data
        
        # Construct Response
        stats = {
            "count": final_total,
            "avg_setting": 0.0,
            "high_confidence": 0.0,
            "probs": {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            "bins_aggregated": final_bins,
            "search_radius": used_radius
        }

        if final_total > 0:
            stats["probs"][1] = (final_c[1] / final_total) * 100.0
            stats["probs"][2] = (final_c[2] / final_total) * 100.0
            stats["probs"][3] = (final_c[3] / final_total) * 100.0
            stats["probs"][4] = (final_c[4] / final_total) * 100.0
            stats["probs"][5] = (final_c[5] / final_total) * 100.0
            stats["probs"][6] = (final_c[6] / final_total) * 100.0
            
            w_sum = sum(s * final_c[s] for s in range(1, 7))
            stats["avg_setting"] = w_sum / final_total
            stats["high_confidence"] = stats["probs"][5] + stats["probs"][6]

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading Matrix Maps (Level 9 S1-S6)...")
    load_maps()
    print("Server Universal (IIIIII 24-byte) Ready on 5001")
    app.run(host='0.0.0.0', port=5001)
