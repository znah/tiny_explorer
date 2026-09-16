import wasmtime
import os
import sys
import struct
import numpy as np
import ctypes

class Cell:
    def __init__(self, exp, idx):
        self._exp = exp
        self._idx = idx
    
    @property
    def name(self):
        return self._exp.read_string(self._exp.exports["wasm_cell_name"](self._exp.store, self._idx))

    @property
    def rects(self):
        """Returns nx4 int32 array of [x1, y1, x2, y2]."""
        arr = self._exp._get_item_array("cell_rects", self._idx, np.int32)
        return arr.reshape(-1, 4) if arr is not None else None

    @property
    def rect2wire(self):
        """Returns array of local wire IDs for each rectangle."""
        return self._exp._get_item_array("cell_rect2wire", self._idx, np.int32)

    def get_bvh(self, layer_idx):
        """Returns nx7 int32 array of [x1, y1, x2, y2, first, count, left]."""
        ptr = self._exp.exports["wasm_cell_bvh_ptr"](self._exp.store, self._idx, layer_idx)
        size = self._exp.exports["wasm_cell_bvh_size"](self._exp.store, self._idx, layer_idx)
        if ptr == 0 or size == 0: return None
        return self._exp._get_mem_view()[ptr:ptr+size].view(np.int32).reshape(-1, 7)

    def __repr__(self):
        return f"Cell({self._idx}: {self.name})"

class Explorer:
    def __init__(self, wasm_path):
        if not os.path.exists(wasm_path):
            raise FileNotFoundError(f"WASM module not found at: {wasm_path}")
            
        self.engine = wasmtime.Engine()
        self.linker = wasmtime.Linker(self.engine)
        self.linker.define_wasi()
        
        self.store = wasmtime.Store(self.engine)
        wasi_config = wasmtime.WasiConfig()
        wasi_config.inherit_stdout()
        wasi_config.inherit_stderr()
        # Pre-open current directory to allow loading GDS files
        wasi_config.preopen_dir(".", ".")
        self.store.set_wasi(wasi_config)
        
        self.module = wasmtime.Module.from_file(self.engine, wasm_path)
        self.instance = self.linker.instantiate(self.store, self.module)
        self.exports = self.instance.exports(self.store)
        
        # Memory access
        self.memory = self.exports["memory"]
        
        # Initialize
        self.init()
        if "wasm_arena_init" in self.exports:
            self.exports["wasm_arena_init"](self.store, 1)

    def init(self):
        """Initializes the WASM module and flushes logs."""
        if "wasm_init" in self.exports:
            self.exports["wasm_init"](self.store)

    def _get_mem_view(self):
        """Returns a zero-copy numpy view of the entire WASM memory."""
        ptr = self.memory.data_ptr(self.store)
        length = self.memory.data_len(self.store)
        return np.ctypeslib.as_array(ptr, shape=(length,))

    def alloc_string(self, s):
        data = s.encode('utf-8') + b'\0'
        if "wasm_malloc" not in self.exports:
            raise AttributeError("WASM module does not export 'wasm_malloc'")
        ptr = self.exports["wasm_malloc"](self.store, len(data))
        mem = self._get_mem_view()
        mem[ptr:ptr+len(data)] = np.frombuffer(data, dtype=np.uint8)
        return ptr

    def free_string(self, ptr):
        if "wasm_free" in self.exports:
            self.exports["wasm_free"](self.store, ptr)

    def load_file(self, path, pdk="sky130A"):
        path_ptr = self.alloc_string(path)
        pdk_ptr = self.alloc_string(pdk)
        try:
            res = self.exports["wasm_load_file"](self.store, path_ptr, pdk_ptr)
            return bool(res)
        finally:
            self.free_string(path_ptr)
            self.free_string(pdk_ptr)

    def process(self):
        if "wasm_process" not in self.exports:
            raise AttributeError("WASM module does not export 'wasm_process'")
        return bool(self.exports["wasm_process"](self.store))

    @property
    def cells(self):
        """Returns list of Cell proxy objects."""
        return [Cell(self, i) for i in range(self.exports["wasm_cell_count"](self.store))]

    @property
    def labeled_wires(self):
        if "wasm_labeledCount" not in self.exports:
            return []
        count = self.exports["wasm_labeledCount"](self.store)
        labels = []
        for i in range(count):
            id_ = self.exports["wasm_circuit_get_labeled_id"](self.store, i)
            name_ptr = self.exports["wasm_circuit_get_labeled_name"](self.store, i)
            name = self.read_string(name_ptr)
            labels.append((id_, name))
        return labels

    def read_string(self, ptr):
        if ptr == 0: return None
        length = self.exports["wasm_strlen"](self.store, ptr)
        if length == 0: return ""
        mem = self._get_mem_view()
        return bytes(mem[ptr:ptr+length]).decode('utf-8')

    @property
    def fets(self):
        """Returns nx5 structured array of FET data."""
        # FET struct layout: uint32 gate, uint32 term[2], uint8 type, [3 bytes padding], int32 instance
        # Total size: 20 bytes
        dtype = np.dtype([
            ('gate', np.uint32),
            ('term', np.uint32, (2,)),
            ('type', np.uint8),
            ('padding', np.uint8, (3,)),
            ('instance', np.int32)
        ])
        return self._get_array("fets", dtype)

    def get_cluster_graph(self, target_wire):
        """Returns a graphviz.Digraph object showing the expanded neighborhood of a wire."""
        try:
            import graphviz
        except ImportError:
            raise ImportError("The 'graphviz' Python module is required for this feature.")

        fets = self.fets
        if fets is None: return None
        
        wire_states = self.wire_data
        net2name = {id_: name for id_, name in self.labeled_wires}
        
        # Pre-build net-to-FET index for fast BFS ($O(N_{fets})$ once instead of $O(N^2)$)
        net_to_fets = {}
        for f in fets:
            for t in f['term']:
                if t not in net_to_fets: net_to_fets[t] = []
                net_to_fets[t].append(f)
        
        cluster_wires = {target_wire}
        q = [target_wire]
        head = 0
        while head < len(q):
            w = q[head]
            head += 1
            # Fast lookup using the index
            for f in net_to_fets.get(w, []):
                peer = f['term'][1] if f['term'][0] == w else f['term'][0]
                if peer != -1 and peer >= 2 and peer not in cluster_wires:
                    cluster_wires.add(peer)
                    q.append(peer)
        
        viz_wires = set(cluster_wires)
        viz_fets = []
        
        for i, f in enumerate(fets):
            t0_in = f['term'][0] in cluster_wires
            t1_in = f['term'][1] in cluster_wires
            g_in = f['gate'] in cluster_wires
            
            if t0_in or t1_in or g_in:
                viz_fets.append((i, f))
                viz_wires.add(f['gate'])
                viz_wires.add(f['term'][0])
                viz_wires.add(f['term'][1])

        dot = graphviz.Digraph('G')
        dot.attr(rankdir='LR')
        dot.attr('node', fontname='sans-serif', fontsize='10')

        for w in sorted(viz_wires):
            label = "VSS" if w == 0 else "VDD" if w == 1 else net2name.get(w, f"w{w}")
            color = "blue" if w == target_wire else "darkgreen" if w in cluster_wires else "black"
            penwidth = "3" if w == target_wire else "2" if w in cluster_wires else "1"
            shape = "plaintext" if w < 2 else "ellipse"
            dot.node(f"w{w}", label=label, color=color, penwidth=penwidth, shape=shape)
        
        for i, fet in viz_fets:
            is_n = (fet['type'] == 1) # FET::N = 1
            g_val = wire_states[fet['gate']] & 1 if fet['gate'] < len(wire_states) else 0
            is_open = (g_val == 1) if is_n else (g_val == 0)
            
            type_str = "N" if is_n else "P"
            color = "green" if is_n else "red"
            penwidth = "3" if is_open else "1"
            
            dot.node(f"f{i}", label=f"{type_str}({fet['instance']})", 
                     shape="box", color=color, width="0.2", height="0.2", penwidth=penwidth)
            
            dot.edge(f"f{i}", f"w{fet['term'][0]}", arrowhead="none", penwidth=penwidth)
            dot.edge(f"f{i}", f"w{fet['term'][1]}", arrowhead="none", penwidth=penwidth)
            if fet['gate'] >= 0:
                dot.edge(f"w{fet['gate']}", f"f{i}", style="dashed")
        
        return dot

    def dump_dot(self, target_wire, filename="dump.dot"):
        """Convenience method to save a cluster graph to a file."""
        dot = self.get_cluster_graph(target_wire)
        if dot:
            dot.save(filename)
            print(f"Dumped wire {target_wire} and its neighborhood to {filename}")

    @property
    def metrics(self):
        metrics = {}
        if "wasm_arena_get_usage" in self.exports:
            metrics["arena_usage_mb"] = self.exports["wasm_arena_get_usage"](self.store) / (1024*1024)
        if "wasm_shortCount" in self.exports:
            metrics["short_count"] = self.exports["wasm_shortCount"](self.store)
        return metrics

    def _get_array(self, name, dtype):
        return self._get_item_array(name, None, dtype)

    def _get_item_array(self, name, idx, dtype):
        ptr_name = f"wasm_{name}_ptr"
        size_name = f"wasm_{name}_size"
        if ptr_name not in self.exports: 
            return None
        args = [self.store]
        if idx is not None: args.append(idx)
        
        ptr = self.exports[ptr_name](*args)
        size_bytes = self.exports[size_name](*args)
        if ptr == 0 or size_bytes == 0: return None
        mem = self._get_mem_view()
        return mem[ptr:ptr+size_bytes].view(dtype)

    @property
    def rects(self):
        """Returns nx5 int32 array of [x1, y1, x2, y2, wire_id, tree_root]."""
        arr = self._get_array("flatRects", np.int32)
        return arr.reshape(-1, 6) if arr is not None else None

    @property
    def wire_data(self):
        """Returns uint8 array of wire states."""
        return self._get_array("wireData", np.uint8)

    @property
    def layer_offsets(self):
        """Returns uint32 array of layer start indices."""
        return self._get_array("flatLayerOffsets", np.uint32)

    @property
    def vga_buffer(self):
        """Returns RGB vga buffer view."""
        arr = self._get_array("vga_buffer", np.uint8)
        if arr is None: return None
        stride = self.exports["wasm_vga_stride"](self.store) if "wasm_vga_stride" in self.exports else 1440
        width = self.exports["wasm_vga_width"](self.store)
        height = self.exports["wasm_vga_height"](self.store)
        return arr.reshape(-1, stride, 3)[:height, :width]


    @property
    def layer_names(self):
        """Returns list of layer names matching the offsets."""
        return [self.read_string(self.exports["wasm_circuit_get_layer_name"](self.store, i)) 
                for i in range(len(self.layer_offsets) - 1)]

    def get_rects_by_layer(self):
        """Returns a list of arrays, one per layer."""
        r = self.rects
        l = self.layer_offsets
        if r is None or l is None: return []
        return [r[l[i]:l[i+1]] for i in range(len(l)-1)]

    def get_layer_bvh(self, layer_idx):
        """Returns nx7 int32 array of [x1, y1, x2, y2, first, count, left]."""
        ptr = self.exports["wasm_layer_bvh_ptr"](self.store, layer_idx)
        size = self.exports["wasm_layer_bvh_size"](self.store, layer_idx)
        if ptr == 0 or size == 0: return None
        return self._get_mem_view()[ptr:ptr+size].view(np.int32).reshape(-1, 7)

    def get_flat_bvh(self, layer_idx):
        """Returns nx7 int32 array of [x1, y1, x2, y2, first, count, left] for full geometry."""
        ptr = self.exports["wasm_flatBVHs_ptr"](self.store, layer_idx)
        size = self.exports["wasm_flatBVHs_size"](self.store, layer_idx)
        if ptr == 0 or size == 0: return None
        return self._get_mem_view()[ptr:ptr+size].view(np.int32).reshape(-1, 7)

    def get_layer_instances(self, layer_idx):
        """Returns nx12 int32 array of [x1, y1, x2, y2, cell_id, inst_id, m00...ty]."""
        ptr = self.exports["wasm_layer_instances_ptr"](self.store, layer_idx)
        size = self.exports["wasm_layer_instances_size"](self.store, layer_idx)
        if ptr == 0 or size == 0: return None
        return self._get_mem_view()[ptr:ptr+size].view(np.int32).reshape(-1, 12)

def main():
    # Attempt to find the WASM module
    wasm_candidates = ["web/explorer.wasm", "zig-out/bin/explorer.wasm", "explorer.wasm"]
    wasm_path = next((p for p in wasm_candidates if os.path.exists(p)), None)
    
    if not wasm_path:
        print("Error: Could not find explorer.wasm. Please build the project first.")
        sys.exit(1)

    print(f"Using WASM module: {wasm_path}")
    exp = Explorer(wasm_path)

    # Test with znah_vga_ca
    gds_file = "gds/09/tt_um_znah_vga_ca.gds"
    if not os.path.exists(gds_file):
        print(f"Error: Could not find GDS file at {gds_file}")
        sys.exit(1)

    print(f"--- Loading {gds_file} ---")
    if exp.load_file(gds_file):
        print("Parsing successful.")
        print("Running connectivity extraction...")
        exp.process()
        
        m = exp.metrics
        if "arena_usage_mb" in m:
            print(f"Memory Usage: {m['arena_usage_mb']:.2f} MB")
        if "short_count" in m:
            print(f"Shorts detected: {m['short_count']}")
        
        # labels = exp.labeled_wires
        # print(f"\nFound {len(labels)} labeled nets:")
        # for id_, name in sorted(labels, key=lambda x: x[1]):
        #     print(f"  Net {id_:>5}: {name}")
        
        r = exp.rects
        print(f"\nExtracted {len(r)} flattened rectangles.")
    else:
        print(f"Error: Failed to load {gds_file}.")

if __name__ == "__main__":
    main()
