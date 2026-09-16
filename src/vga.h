#pragma once

#include <map>
#include "fetsim.h"
#include "extractor.h"

struct VGASimulator {
    Circuit& circuit;
    
    std::map<std::string, int> name2id;
    int clk = -1, rst_n = -1, ena = -1;
    int in_pins[8] = {-1,-1,-1,-1,-1,-1,-1,-1}, out_pins[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    
    const int max_width = 1440, max_height = 720;
    int width = 720, height = 480;
    int ray_x = 0, ray_y = 0;
    bool last_hsync=false, last_vsync=false;
    std::vector<uint8_t> vga_buffer;

    VGASimulator(Circuit& c) : circuit(c) {
        vga_buffer.assign(max_width * max_height * 3, 0);
    }

    void init(const std::vector<CircuitExtractor::LabeledWire>& labels) {
        name2id.clear();
        for (const auto& lw : labels) {
            name2id[lw.name] = lw.id;
        }
        // Find essential pins
        clk = getPin("clk");
        rst_n = getPin("rst_n");
        ena = getPin("ena");
        
        for (int i = 0; i < 8; ++i) {
            out_pins[i] = getPin("uo_out[" + std::to_string(i) + "]");
            in_pins[i] = getPin("ui_in[" + std::to_string(i) + "]");
        }
    }

    void reset() {
        if (!isValid()) return;
        if (ena != -1) circuit.set_signal(ena, 1);
        circuit.set_signal(rst_n, 0);
        // TT-specific reset behavior
        if (in_pins[1] != -1) circuit.set_signal(in_pins[1], 1);
        if (in_pins[4] != -1) circuit.set_signal(in_pins[4], 1);
        for (int i = 0; i < 10; ++i) {
            circuit.set_signal(clk, 0); circuit.settle(200);
            circuit.set_signal(clk, 1); circuit.settle(200);
            vga_tick();
        }
        if (in_pins[1] != -1) circuit.set_signal(in_pins[1], 0);
        if (in_pins[4] != -1) circuit.set_signal(in_pins[4], 0);
        circuit.set_signal(clk, 0); circuit.settle(200);
        circuit.set_signal(rst_n, 1); circuit.settle(200);
        ray_x = 0; ray_y = 0;
        last_hsync = false; last_vsync = false;
    }

    bool isValid() const { return clk != -1 && rst_n != -1; }

    void run(int max_ticks = 1000000) {
        if (!isValid()) return;
        
        printf("VGA Sim: clk=%d, rst_n=%d, outputs=[", clk, rst_n);
        for(int i=0; i<8; ++i) printf("%d%s", out_pins[i], i==7 ? "]\n" : ",");

        printf("Resetting...\n");
        reset();
        
        printf("Running simulation...\n");
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < max_ticks; ++i) {
            circuit.set_signal(clk, 0); circuit.settle(200);
            circuit.set_signal(clk, 1); circuit.settle(200);
            vga_tick();
            if ((i + 1) % 10000 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                uint8_t out = 0;
                for (int j=0; j<8; j++) {
                    if (out_pins[j] != -1 && (circuit.wire_data[out_pins[j]] & Circuit::V_MASK)) out |= (1 << j);
                }
                printf("  Tick %d (%.0f px/s), ray at %d,%d, out=%02x  shorts=%d\n",
                    i + 1, (i + 1) / elapsed, ray_x, ray_y, out, circuit.short_count);
                savePPM("vga.ppm");
            }
        }
        savePPM("vga.ppm");
    }

    int getPin(const std::string& name) {
        return name2id.count(name) ? name2id.at(name) : -1;
    }

    void settle(int max_waves=200) {
        circuit.settle(max_waves);
    }

    void vga_tick() {
        auto val = [&](int i) { return (out_pins[i] != -1) && (circuit.wire_data[out_pins[i]] & Circuit::V_MASK); };
        bool r1 = val(0), g1 = val(1), b1 = val(2), vsync = val(3);
        bool r0 = val(4), g0 = val(5), b0 = val(6), hsync = val(7);

        if (ray_x < max_width && ray_y < max_height) {
            int idx = (ray_y * max_width + ray_x) * 3;
            vga_buffer[idx + 0] = (r1 * 2 + r0) * 85;
            vga_buffer[idx + 1] = (g1 * 2 + g0) * 85;
            vga_buffer[idx + 2] = (b1 * 2 + b0) * 85;
        }
        ray_x++;

        if (!last_hsync && hsync) { 
            width = std::min(std::max(ray_x, width), max_width);
            ray_x = 0; ray_y++; 
        }
        if (!last_vsync && vsync) { 
            height = std::min(std::max(ray_y, height), max_height); 
            ray_y = 0;
        }

        last_hsync = hsync;
        last_vsync = vsync;
    }

    void savePPM(const char* filename) {
        FILE* f = fopen(filename, "wb");
        if (!f) return;
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        for (int i=0; i<height; ++i) {
            fwrite(vga_buffer.data() + i*max_width*3, 1, width*3, f);    
        }
        fclose(f);
    }
};
