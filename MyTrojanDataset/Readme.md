1. This dataset has 131 Trojan-free circuits (in `TjFree.zip`) sourced from [TrustHub](https://trust-hub.org/#/benchmarks/chip-level-trojan)  and [Pyverilog](https://github.com/PyHDI/Pyverilog). 

2. This dataset has 50 Trojan-ed samples (in `TjIn.zip`) sourced from [TrustHub](https://trust-hub.org/#/benchmarks/chip-level-trojan) and [MyTrojans](https://github.com/sumandeb003/Ariane_Trojans_for_Pyverilog).

3. Not all the samples in TrustHub could be converted to graphs using the Pyverilog-based circuit-to-graph conversion mechanism of the [HW2VEC](https://github.com/AICPS/hw2vec) tool. This is because:

i. Some are VHDL RTLs and some are layouts. Hence, incompatible with Pyverilog.
ii. Some are gate-level netlists that have been synthesized for FPGA. So, the netlist contains gates that can't be recognised by Pyverilog.


4. The Trojan-free circuits in my dataset are:

`adder4bit_1`  
`c432-NC2360`  
`c499-NS1870`  
`c880-NS3080`     
`generate`
`adder4bit_2`  
`c432-NR1400`  
`c499-NS2510`  
`c880-NS4360`     
`generate_instance`
`adder4bit_3`  
`c432-NR1720`  
`c499-RN1280`  
`c880-RN1280`     
`instance_array`
`adder4bit_4`  
`c432-NR2360`  
`c499-RN320`  
`c880-RN2560`     
`instance_empty_params`
`adder4bit_5`  
`c432-NS1400`  
`c499-RN640`  
`c880-RN320`      
`led_main`
`adder4bit_6`  
`c432-NS1720`  
`c499-SL1280`  
`c880-RN640`      
`partial`
`adder4bit_7`  
`c432-NS2360`  
`c499-SL320`   
`c880-SL1280`     
`partselect_assign`
`AES`          
`c432-RN1280`  
`c499-SL640`   
`c880-SL2560`     
`PIC16F84`
`bcdToseg_1`   
`c432-RN320`   
`c880-CS1280`  
`c880-SL320`      
`primitive`
`bcdToseg_2`   
`c432-RN640`   
`c880-CS2560`  
`c880-SL640`      
`ptr_clock_reset`
`bcdToseg_3`   
`c432-SL1280`  
`c880-CS320`   
`case`            
`ram`
`bcdToseg_4`   
`c432-SL320`   
`c880-CS640`   
`case_in_func`    
`RC5`
`bcdToseg_5`   
`c432-SL640`   
`c880-CY1070`  
`casex`           
`RC6`
`bcdToseg_6`   
`c499-CS1280`  
`c880-CY2030`  
`count`           
`reset`
`bcdToseg_7`   
`c499-CS320`   
`c880-CY310`   
`decimal`        
`RS232`
`bcdToseg_8`  
`c499-CS640`  
`c880-CY3880` 
`decimal_signed`  
`signed_task`
`blocking`     
`c499-CY1040`  
`c880-CY590`   
`decimal_width`   
`spi_master`
`c432-BE280`   
`c499-CY2060`  
`c880-NC2120`  
`deepcase`        
`statemachine`
`c432-CS1280`  
`c499-CY270`  
`c880-NC2440`  
`delay`          
`supply`
`c432-CS320`   
`c499-CY520`   
`c880-NC3080`  
`det_1011`        
`syncRAM`
`c432-CS640`   
`c499-NC1550`  
`c880-NC4360` 
`encoder8to3_1`  
`vectoradd`
`c432-CY1020`  
`c499-NC1870` 
`c880-NR2120`
`encoder8to3_2` 
`vga`
`c432-CY2000`  
`c499-NC2510` 
`c880-NR2440`
`encoder8to3_3`  
`xtea`
`c432-CY290`   
`c499-NR1550` 
`c880-NR3080` 
`encoder8to3_4`
`c432-CY530`
`c499-NR1870`
`c880-NR4360`
`encoder8to3_5`
`c432-NC1400`
`c499-NR2510`
`c880-NS2120`
`encoder8to3_6`
`c432-NC1720`
`c499-NS1550`
`c880-NS2440`
`function`



5. The Trojan-ed circuits in my dataset are:

`AES-T100`   `AES-T1300`  `AES-T1700`  `AES-T2000`  `AES-T2500`  `AES-T300`  `AES-T700`      `PIC16F84-T100`  `RS232-T100`   `RS232-T2300`  `RS232-T500`  `RS232-T900`     `vectoradd_TP3`
`AES-T1000`  `AES-T1400`  `AES-T1800`  `AES-T2100`  `AES-T2600`  `AES-T400`  `AES-T800`      `PIC16F84-T200`  `RS232-T200`   `RS232-T2400`  `RS232-T600`  `RS232-T901`     `wb_conmax-T100`
`AES-T1100`  `AES-T1500`  `AES-T1900`  `AES-T2300`  `AES-T2700`  `AES-T500`  `AES-T900`      `PIC16F84-T300`  `RS232-T2100`  `RS232-T300`   `RS232-T700`  `vectoradd_TP1`
`AES-T1200`  `AES-T1600`  `AES-T200`   `AES-T2400`  `AES-T2800`  `AES-T600`  `blocking-TP1`  `PIC16F84-T400`  `RS232-T2200`  `RS232-T400`   `RS232-T800`  `vectoradd_TP2`

6. I am currently debugging few more RTLs to include them in my dataset.
