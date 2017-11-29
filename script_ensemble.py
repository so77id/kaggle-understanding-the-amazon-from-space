import numpy as np
import sys



def main(args):
	net_outputs_1,net_outputs_2,net_outputs_3= args[1], args[2], args[3]
	output_preds_file = args[4]
	threshold = float(args[5])
	numtags=17
	ensemble_file=open(output_preds_file,'w')
	ensemble_file_lines=[]
	ensemble_soft_prediction=[]

	labels = ["agriculture","artisinal_mine","bare_ground","blooming","blow_down","clear",
	"cloudy","conventional_mine","cultivation","habitation","haze","partly_cloudy","primary"
	,"road","selective_logging","slash_burn","water"]

	# open files
	net1 = open(net_outputs_1,'r')
	net2 = open(net_outputs_2,'r')
	net3 = open(net_outputs_3,'r')

	lines_net1 = net1.readlines()
	lines_net2 = net2.readlines()
	lines_net3 = net3.readlines()

	for idx in range(len(lines_net1)):
		vector1,vector2,vector3 = lines_net1[idx].split(',')[1],lines_net2[idx].split(',')[1],lines_net3[idx].split(',')[1]
		lines_1=np.array(vector1.split(" "),dtype=np.float)
		lines_2=np.array(vector2.split(" "),dtype=np.float)
		lines_3=np.array(vector3.split(" "),dtype=np.float)


		test_Y_1 = np.where(lines_1>=threshold)[0]
		test_Y_2 = np.where(lines_2>=threshold)[0]
		test_Y_3 = np.where(lines_3>=threshold)[0]

		final_Y=[]
		tags1,tags2,tags3 = np.zeros(numtags),np.zeros(numtags),np.zeros(numtags)
		tags1[test_Y_1]=1
		tags2[test_Y_2]=1
		tags3[test_Y_3]=1
		
		for index in range(len(tags1)):
			preds=[tags1[index],tags2[index],tags3[index]]
			ensemble_tags=np.argmax(np.bincount(preds))
	        	ensemble_soft_prediction.append(ensemble_tags)
		ensemble_soft_prediction = np.array(ensemble_soft_prediction)
		tags=np.where(ensemble_soft_prediction == 1)[0]
		pred_labels=[labels[tag] for tag in tags]
		
		filename= lines_net1[idx].split(',')[0]
		ensemble_file_lines.append(filename +','+ ','.join(pred_labels) + '\n')
		ensemble_soft_prediction=[]

	ensemble_file.writelines(ensemble_file_lines)
	net1.close()
	net2.close()
	net3.close()
	ensemble_file.close()


if __name__ == '__main__':
	main(sys.argv)


