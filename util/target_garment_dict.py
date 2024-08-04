jin_dict={i:"jin_"+str(i).zfill(2)+"_white_bg.jpg" for i in range(16)}
lab_dict={i:"lab_"+str(i).zfill(2)+"_white_bg.jpg" for i+16 in range(9)}
target_garment_dict=jin_dict.copy()
target_garment_dict.update(lab_dict)