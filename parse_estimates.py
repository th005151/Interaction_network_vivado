import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import hls4ml
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)
directory = './'
#hls4ml.report.vivado_report.read_vivado_report(directory, full_report=False)

def read_vivado_report(dsp, bram, ff, lut, latency, ii, hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/build_prj.tcl'):
        prj_dir, top_func_name = _parse_build_script(hls_dir + '/build_prj.tcl')

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return
    
    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
        return
    
    solutions = _find_solutions(sln_dir)
    print('Found {} solution(s) in {}.'.format(len(solutions), sln_dir))

    for sln in solutions:
        print('Reports for solution "{}":\n'.format(sln))
        _find_reports(dsp, bram, ff, lut, latency, ii, sln_dir + '/' + sln, top_func_name, full_report)

def _parse_build_script(script_path):
    prj_dir = None
    top_func_name = None

    with open(script_path, 'r') as f:
        for line in f.readlines():
            if 'open_project' in line:
                prj_dir = line.split()[-1]
            elif 'set_top' in line:
                top_func_name = line.split()[-1]
    
    return prj_dir, top_func_name

def _find_solutions(sln_dir):
    solutions = []

    if os.path.isfile(sln_dir + '/vivado_hls.app'):
        with open(sln_dir + '/vivado_hls.app') as f:
            # Get rid of namespaces (workaround to support two types of vivado_hls.app files)
            xmlstring = re.sub(' xmlns="[^"]+"', '', f.read(), count=1)

        root = ET.fromstring(xmlstring)
        for sln_tag in root.findall('solutions/solution'):
            sln_name = sln_tag.get('name')
            if sln_name is not None and os.path.isdir(sln_dir + '/' + sln_name):
                solutions.append(sln_name)
    
    return solutions

def _find_reports(dsp, bram, ff, lut, latency, ii, sln_dir, top_func_name, full_report=False):
    #csim_file = sln_dir + '/csim/report/{}_csim.log'.format(top_func_name)
    #if os.path.isfile(csim_file):
    #    _show_csim_report(csim_file)
    #else:
    #    print('C simulation report not found.')
    
    syn_file = sln_dir + '/syn/report/{}_csynth.rpt'.format(top_func_name)
    if os.path.isfile(syn_file):
        _show_synth_report(dsp, bram, ff, lut, latency, ii, syn_file, full_report)
    else:
        print('Synthesis report not found.')

def _show_csim_report(csim_file):
    with open(csim_file, 'r') as f:
        print('C SIMULATION RESULT:')
        print(f.read())

def _show_synth_report(dsp, bram, ff, lut, latency, ii, synth_file, full_report=False):
    #temp_latency = 0
    #temp_dsp = 0
    c_edge = 0
    c_node = 0
    temp_latency = 0
    temp_dsp = 0
    temp_ii = 0
    with open(synth_file, 'r') as f:
        print('SYNTHESIS REPORT:')
        for line in f.readlines()[2:]:
            if not full_report and '* DSP48' in line:
                break
            #print(line, end = '')
            """
            # count estimates for IN
            if 'IN_edge_module' in line and c_edge == 0:
                temp_latency += int(line[68:77].strip())
                temp_ii += int(line[109:115].strip())
                c_edge += 1
            elif 'IN_node_module' in line and c_node == 0:
                temp_latency += int(line[68:77].strip())
                temp_ii += int(line[109:115].strip())
                c_node += 1
            elif 'IN_edge_module' in line and c_edge == 1:
                temp_dsp += int(line[64:71].strip())
                c_edge += 1
            elif 'IN_node_module' in line and c_node == 1:
                temp_dsp += int(line[64:71].strip())
                c_node += 1
        dsp.append(temp_dsp)
        latency.append(temp_latency)
        ii.append(temp_ii)
            """
            # count estimates for top project
            if 'Latency' in line and temp_latency == 0:
                temp_latency += 1
            elif 'DSP48E' in line and temp_dsp == 0:
                temp_dsp += 1
            if temp_latency == 7:
                latency.append(int(line[15:24].strip()))
                ii.append(int(line[53:58].strip()))
                temp_latency += 1
            elif temp_latency > 0 and temp_latency < 8:
                temp_latency += 1
            elif temp_dsp == 11:
                dsp.append(int(line[33:40].strip()))
                bram.append(int(line[23:32].strip()))
                ff.append(int(line[41:50].strip()))
                lut.append(int(line[51:59].strip()))
                temp_dsp += 1
            elif temp_dsp > 0 and temp_dsp < 12:
                temp_dsp += 1
            else:
                continue
        

DSP = []
BRAM = []
FF = []
LUT = []
Latency = []
II = []
Reuse = [1, 2, 4, 8, 16]
Precision = [10, 12, 16, 20]
maxDSP = 5520
maxBRAM = 4320
maxFF = 1326720
maxLUT = 663360
read_vivado_report(DSP, BRAM, FF, LUT, Latency, II, directory, full_report=False)
print("DSP: ", DSP)
print("BRAM: ", BRAM)
print("FF: ", FF)
print("LUT: ", LUT)
print("Latency: ", Latency)
print("II: ", II)
"""
plt.figure(figsize=(6,6))
plt.plot(Reuse, DSP, '-o', lw=2)
plt.plot([min(Reuse), max(Reuse)], [maxDSP, maxDSP], ls='--', lw=2, color='gray')
plt.xlabel('Reuse factor')
plt.ylabel('DSPs')
#plt.title('DSPs vs Reuse factor')
plt.savefig('DSP_vs_RF.pdf')

plt.figure(figsize=(6,6))
plt.plot(Reuse, BRAM, '-o', lw=2)
plt.plot([min(Reuse), max(Reuse)], [maxBRAM, maxBRAM], ls='--', lw=2, color='gray')
plt.xlabel('Reuse factor')
plt.ylabel('BRAM')
#plt.title('BRAMs vs Reuse factor')
plt.tight_layout()
plt.savefig('BRAM_vs_RF.pdf')

plt.figure(figsize=(6,6))
plt.plot(Reuse, FF, '-o', lw=2)
plt.plot([min(Reuse), max(Reuse)], [maxFF, maxFF], ls='--', lw=2, color='gray')
plt.xlabel('Reuse factor')
plt.ylabel('FFs')
#plt.title('Flip-Flops vs Reuse factor')
plt.tight_layout()
plt.savefig('FF_vs_RF.pdf')

plt.figure(figsize=(6,6))
plt.plot(Reuse, LUT, '-o', lw=2)
plt.plot([min(Reuse), max(Reuse)], [maxLUT, maxLUT], ls='--', lw=2, color='gray')
plt.xlabel('Reuse factor')
plt.ylabel('LUTs')
#plt.title('LUTs vs Reuse factor')
plt.tight_layout()
plt.savefig('LUT_vs_RF.pdf')
"""
plt.figure(figsize=(6,5))
plt.plot(Reuse, 100*np.array(DSP)/maxDSP, '-o', lw=2, label='DSP')
plt.plot(Reuse, 100*np.array(LUT)/maxLUT, '-o', lw=2, label='LUT')
plt.plot(Reuse, 100*np.array(BRAM)/maxBRAM, '-o', lw=2, label='BRAM')
plt.plot(Reuse, 100*np.array(FF)/maxFF, '-o', lw=2, label='FF')
plt.ylim(-10,450)
plt.xlabel('Reuse factor')
plt.ylabel('Usage [%]')
#plt.title('Resources vs Reuse factor')
plt.legend()
plt.legend(title = '28 nodes, 37 edges', loc='upper center')
plt.tight_layout()
plt.savefig('Resources_vs_RF.pdf')

plt.figure(figsize=(6,5))
plt.plot(Reuse, Latency, '-o', lw=2)
plt.ylim(150,200)
plt.xlabel('Reuse factor')
plt.ylabel('Latency [cycles]')
#plt.title('Latency vs Reuse factor')
plt.legend(title = '28 nodes, 37 edges')
plt.tight_layout()
plt.savefig('Latency_vs_RF.pdf')

plt.figure(figsize=(6,6))
plt.plot(Reuse, II, '-o', lw=2)
plt.xlabel('Reuse factor')
plt.ylabel('II [cycles]')
#plt.title('Initiation Interval vs Reuse factor')
plt.tight_layout()
plt.savefig('II_vs_RF.pdf')
"""
plt.figure(figsize=(6,6))
plt.plot(Precision, DSP, '-o', lw=2)
plt.plot([min(Precision), max(Precision)], [maxDSP, maxDSP], ls='--', lw=2, color='gray')
plt.xlabel('Total bit precision')
plt.ylabel('DSPs')
#plt.title('DSPs vs Bit Precision')
#plt.semilogy()
plt.savefig('DSP_vs_BP.pdf')

plt.figure(figsize=(6,6))
plt.plot(Precision, BRAM, '-o', lw=2)
plt.plot([min(Precision), max(Precision)], [maxBRAM, maxBRAM], ls='--', lw=2, color='gray')
plt.xlabel('Total bit precision')
plt.ylabel('BRAM')
#plt.title('BRAMs vs Bit Precision')
plt.tight_layout()
plt.savefig('BRAM_vs_BP.pdf')

plt.figure(figsize=(6,6))
plt.plot(Precision, FF, '-o', lw=2)
plt.xlabel('Total bit precision')
plt.ylabel('FFs')
#plt.title('Flip-Flops vs Bit Precision')
plt.tight_layout()
plt.savefig('FF_vs_BP.pdf')

plt.figure(figsize=(6,6))
plt.plot(Precision, LUT, '-o', lw=2)
plt.plot([min(Precision), max(Precision)], [maxLUT, maxLUT], ls='--', lw=2, color='gray')
plt.xlabel('Total bit precision')
plt.ylabel('LUTs')
#plt.title('LUTs vs Bit Precision')
plt.tight_layout()
plt.savefig('LUT_vs_BP.pdf')


plt.figure(figsize=(6,5))
plt.plot(Precision, 100*np.array(DSP)/maxDSP, '-o', lw=2, label='DSP')
plt.plot(Precision, 100*np.array(LUT)/maxLUT, '-o', lw=2, label='LUT')
plt.plot(Precision, 100*np.array(BRAM)/maxBRAM, '-o', lw=2, label='BRAM')
plt.plot(Precision, 100*np.array(FF)/maxFF, '-o', lw=2, label='FF')
plt.ylim(-10,200)
plt.xlabel('Total bit precision')
plt.ylabel('Usage [%]')
#plt.title('LUTs vs Bit Precision')
plt.legend()
plt.legend(title = '28 nodes, 37 edges', loc='upper right')
plt.tight_layout()
plt.savefig('Resources_vs_BP.pdf')

plt.figure(figsize=(6,5))
plt.plot(Precision, Latency, '-o', lw=2)
plt.ylim(150,200)
plt.xlabel('Total bit precision')
plt.ylabel('Latency [cycles]')
#plt.title('Latency vs Bit Precision')
plt.legend(title = '28 nodes, 37 edges')
plt.tight_layout()
plt.savefig('Latency_vs_BP.pdf')

plt.figure(figsize=(6,6))
plt.plot(Precision, II, '-o', lw=2)
plt.xlabel('Total bit precision')
plt.ylabel('Initiation interval [cycles]')
#plt.title('Initiation Interval vs Bit Precision')
plt.tight_layout()
plt.savefig('II_vs_BP.pdf')
"""
