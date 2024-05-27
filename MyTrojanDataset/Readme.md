1. This dataset has 149 Trojan-free circuits (in `TjFree.zip`) sourced from [TrustHub](https://trust-hub.org/#/benchmarks/chip-level-trojan)  and [Pyverilog](https://github.com/PyHDI/Pyverilog). 

2. This dataset has 2946 Trojan-ed samples (in `TjIn.zip`) sourced from [TrustHub](https://trust-hub.org/#/benchmarks/chip-level-trojan) and [MyTrojans](https://github.com/sumandeb003/Ariane_Trojans_for_Pyverilog).

3. Not all the samples in TrustHub could be converted to graphs using the Pyverilog-based circuit-to-graph conversion mechanism of the [HW2VEC](https://github.com/AICPS/hw2vec) tool. This is because:

i. Some are VHDL RTLs and some are layouts. Hence, incompatible with Pyverilog.

ii. Some are gate-level netlists that have been synthesized for FPGA. So, the netlist contains gates that can't be recognised by Pyverilog.

