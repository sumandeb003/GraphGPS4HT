import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.hw2graph import *
from hw2vec.config import Config
from pathlib import Path

def delete_all_ptfiles_in_all_subdirectories(directory_path):
	for subdir in os.listdir(directory_path):
		subdir_path = os.path.join(directory_path,subdir)
		for circuit in os.listdir(subdir_path):
			if circuit.endswith('.pt'):
				circuit_path = os.path.join(subdir_path,circuit)
				print('Deleting: ', circuit_path)
				os.remove(circuit_path)


def TrustHub_to_graph(cfg, hw_design_dir_path):
    hw2graph = HW2GRAPH(cfg)
    hw_design_path = hw2graph.preprocess(hw_design_dir_path) #flatten all .v files to one .v file, remove comments, remove underscores, rename as topModule.v
    hardware_nxgraph = hw2graph.process(hw_design_path) #generate AST/DFG (JSON format) of the topModule.v
    data_proc = DataProcessor(cfg)
    data_proc.process(hardware_nxgraph)#normalize the graph and create node-feature vectors X and adjacency matrix A
    TROJAN = 1
    NON_TROJAN = 0
    data = data_proc.get_graphs()
    print (f'TrustHub_to_graph: data - {data}')
    if "TjFree" == str(hw_design_path).split("/")[-3]:
    	data[0].label = NON_TROJAN
    else:
    	data[0].label = TROJAN
    save_path = os.path.join(hw_design_dir_path, f'{cfg.graph_type}.pt')
    torch.save(data[0], save_path)

if __name__ == '__main__':
    #redirect output messages - print statements - to a log file
    try:
    	os.remove('TrustHub2graph.log')
    except:
    	pass
    log = open('TrustHub2graph.log', 'a')
    sys.stdout = log
    #delete all previous .pt files in all sub-directories of directory_path
    directory_path = '../../Downloads/TrustHub/4GraphGPS/TjFree' #like, path to TjFree or TjIn
    delete_all_ptfiles_in_all_subdirectories(directory_path)
    #create new .pt files
    cfg = Config(sys.argv[1:]) #because of the following 4 lines, the statement 'cfg = Config(sys.argv[1:]) has no use
    #iterate through all folders in TjFree
    for subdir in os.listdir(directory_path):
    	subdir_path = Path(os.path.join(directory_path,subdir))
    	print('===================================================================')
    	print (f'TrustHub_to_graph: circuit - {subdir_path}')
    	#do both AST and DFG
    	for graphtype in ['AST', 'DFG']:
    		cfg.graph_type = graphtype
    		print (f'TrustHub_to_graph: graphtype - {cfg.graph_type}')
    		try:
    			TrustHub_to_graph(cfg, subdir_path)
    		except Exception as error:
    			print("ERROR:	", type(error).__name__, "–", error)
    	print('===================================================================')

    log.close()
