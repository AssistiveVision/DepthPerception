def closeness_warning(depth_image,x,y,w,h):
    print(depth_image.shape)
    cropped_depth_map=depth_image[0, int(y/2) : int((y+h)/2), int(x/2):int((x+w)/2) , : ]
    print(cropped_depth_map.shape)
    max_pixel=0
    for i in range(cropped_depth_map.shape[0]):
        for j in range(cropped_depth_map.shape[1]):
            if(cropped_depth_map[i][j][0]>max_pixel):
                max_pixel=cropped_depth_map[i][j][0]
    
    
    if max_pixel in range(0,int(255/3)):
        # (0,255/3)
        return -1 #"very near" 
    elif max_pixel in range(int(255/3),int(2*255/3)):
        # (255/3,2*255/3)
        return 0 #"near"
    else:
        # (2*255/3,255)
        return 1 #"far away"
    