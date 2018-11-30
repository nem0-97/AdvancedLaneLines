import numpy as np
import cv2
import os
import shutil

#TODO: Refactor some code maybe
#TODO: Maybe change getBaseXs so it doesn't sum up the pixels within a certain distance from bottom of img into the histogram
    #(to get rid of the car's hood) probably could still tune thresholds more or chose different channels to apply thesholds to
#TODO: Calculate more accurate pixel to meter conversion ratios

def calibrate(calDir,nx=9,ny=6,display=False):#takes string for directory containing images to calibrate with and checkerboard sizes
    '''Calibrate using a directory of checkerboard calibration images'''
    '''with nx=9 and ny=6 17 cal images work'''
    objpoints=[]
    imgpoints=[]

    objp=np.zeros((ny*nx,3),np.float32)
    objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    filenames=os.listdir(calDir)

    for cal in filenames:
        absImgPath=os.path.join(os.getcwd(),calDir,cal)
        img=cv2.imread(absImgPath)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret,corners=cv2.findChessboardCorners(gray,(nx,ny),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

            outImg=cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            if display:
                cv2.imshow(cal,outImg)
                while True:
                    if cv2.waitKey(1) & 0xFF==ord('q'):#when they hit 'q' close the image on screen and move to next one
                        break
                cv2.destroyAllWindows()
    #make mtx and dist global(needed for undistort)
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def warp(img,top=True):#top is if trying to go to top down view or not
    '''get top down view of lane lines'''
    #1)get 4 points on same plane as lane lines(on the road)
    srcImg=np.copy(img)
    src=np.float32([[490,2*img.shape[0]/3],[810,2*img.shape[0]/3],[1250,img.shape[0]],[40,img.shape[0]]])

    #2)decide where you want them to move to
    dst=np.float32([[0,0],[img.shape[1],0],[1250,img.shape[0]],[40,img.shape[0]]])

    if top:
        perMatr=cv2.getPerspectiveTransform(src,dst)
    else:
        perMatr=cv2.getPerspectiveTransform(dst,src)#undo perspective transform

    #3)perform perspective transform using proper matrix
    return cv2.warpPerspective(img,perMatr,(img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

def binary(img,color=False,s_thresh=(170, 255),sx_thresh=(30, 100)):
    '''Create binary image by apllying thresholds, color boolean indicates wether to indicate which threshold identified it'''
    img = np.copy(img)
    # Convert to HLS color space and separate the H channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel, BGR, B is all 0, G is what was kept after x gradient threshold, R is the ones kept from S channel threshold
    if color:
       binary=np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))*255
    else:
    #Plain black or white(doesn't indicate which threshold)
        binary=np.zeros_like(sxbinary)
        binary[(s_binary == 1) | (sxbinary == 1)] = 1
        binary=np.dstack((binary, binary, binary))*255

    return binary

def getBaseXs(img):
    '''find x value with max number of white pixels in right and left half of img return them'''
    #get sum of all pixel values, for pixels with each x value, across the bottom half of image
    histogram=np.sum(img[img.shape[0]//2:,:], axis=0)
    #find the x value with the max whiteness value and add it too minX to get its actual pixel x value in image
    midpoint=np.int(histogram.shape[0]//2)
    #since histogram is 1280X3 shaped array use axis=0 which returns size 3 array take first value(all 3 are same)
    x1=np.argmax(histogram[:midpoint],axis=0)[0]
    x2=np.argmax(histogram[midpoint:],axis=0)[0]+midpoint
    return x1,x2

def slidingWindows(img,numWins=9,margin=45,minpix=50):
    '''Iterate across height of image, with numWins windows, get all the pixels that are white in each window and add them to proper lane line'''
    winH=np.int(img.shape[0]//numWins)

    #get nonzero pixels and break up coords
    nonzero=img.nonzero()
    nonzeroYs=np.array(nonzero[0])
    nonzeroXs=np.array(nonzero[1])

    #starting points
    curLeftX,curRightX=getBaseXs(img)

    #to store indices found for each lane line
    left_lane_inds=[]
    right_lane_inds=[]

    #run windows
    for win in range(numWins):
        #window bounds
        minY=img.shape[0]-(win+1)*winH
        maxY=img.shape[0]-win*winH
        minLeftX=curLeftX-margin
        maxLeftX=curLeftX+margin
        minRightX=curRightX-margin
        maxRightX=curRightX+margin

        #nonzero pixels in window
        good_left_inds=((nonzeroYs>=minY)&(nonzeroYs<maxY) & (nonzeroXs>=minLeftX)&(nonzeroXs<maxLeftX)).nonzero()[0]
        good_right_inds=((nonzeroYs>=minY)&(nonzeroYs<maxY) & (nonzeroXs>=minRightX)&(nonzeroXs<maxRightX)).nonzero()[0]

        #keep track of good indices and which side they're for
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if found>minpix pixels recenter to their mean x position
        if len(good_left_inds)>minpix:
            curLeftX=np.int(np.mean(nonzeroXs[good_left_inds]))
        if len(good_right_inds)>minpix:
            curRightX=np.int(np.mean(nonzeroXs[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds=np.concatenate(left_lane_inds)
    right_lane_inds=np.concatenate(right_lane_inds)

    #return left and right line pixels' x and y positions
    return nonzeroXs[left_lane_inds],nonzeroYs[left_lane_inds],nonzeroXs[right_lane_inds],nonzeroYs[right_lane_inds]

def searchFromPrior(img,leftPoly,rightPoly,margin=45):
    '''given 2 polynomials, return coords of white pixels within a margin of each polynomial'''
    #get nonzero pixels and break up coords
    nonzero=img.nonzero()
    nonzeroYs=np.array(nonzero[0])
    nonzeroXs=np.array(nonzero[1])

    #nonzero pixels within margin of error from polynomial they should fit
    left_lane_inds=((nonzeroXs>(leftPoly[0]*(nonzeroYs**2)+leftPoly[1]*nonzeroYs+leftPoly[2]-margin)) & (nonzeroXs<(leftPoly[0]*(nonzeroYs**2)+leftPoly[1]*nonzeroYs+leftPoly[2]+margin)))
    right_lane_inds=((nonzeroXs>(rightPoly[0]*(nonzeroYs**2)+rightPoly[1]*nonzeroYs+rightPoly[2]-margin)) & (nonzeroXs<(rightPoly[0]*(nonzeroYs**2)+rightPoly[1]*nonzeroYs+rightPoly[2]+margin)))

    #return left and right line pixels' x and y positions
    return nonzeroXs[left_lane_inds],nonzeroYs[left_lane_inds],nonzeroXs[right_lane_inds],nonzeroYs[right_lane_inds]

def getPoly(img,leftPoly=None,rightPoly=None,getImg=False):
    '''find polynomial to fit each lane line and draw it onto image, also color pixels used to come up with each polynomial'''
    if (leftPoly is None) or (rightPoly is None):
        leftXs,leftYs,rightXs,rightYs=slidingWindows(img)
    else:
        leftXs,leftYs,rightXs,rightYs=searchFromPrior(img,leftPoly,rightPoly)

    #get polynomial(2nd degree) that fits pixels chosen for each lane line(y pixel values are input to polynomial and x pixel coords are output)
    leftPoly=np.polyfit(leftYs,leftXs,2)
    rightPoly=np.polyfit(rightYs,rightXs,2)

    if getImg:
        #get points along the polynomials to plot
        plotYs=np.linspace(0,img.shape[0]-1,img.shape[0])
        leftPolyXs=leftPoly[0]*plotYs**2+leftPoly[1]*plotYs+leftPoly[2]
        rightPolyXs=rightPoly[0]*plotYs**2+rightPoly[1]*plotYs+rightPoly[2]

        #color in pixels used to find polynomial fo each lane line
        img[leftYs,leftXs]=[255,0,0] #blue
        img[rightYs,rightXs]=[0,0,255] #red

        #draw polynomials for both lane lines onto img
        plotYs=plotYs.astype(int)
        leftPolyXs=leftPolyXs.astype(int)
        rightPolyXs=rightPolyXs.astype(int)


        img[plotYs,leftPolyXs]=[0,255,0]#green
        img[plotYs,rightPolyXs]=[0,255,0]
        #if want to see visualization
        return img,leftPoly,rightPoly

    #return coeffs for the 2 2nd degree polynomials
    return leftPoly,rightPoly

def distFromCenter(imgH,imgW,leftPoly,rightPoly,mppx=(3.7/700)):
    '''Vehicle off center(absolute value of(average of each polynomial when plug in height of img)-(width of img //2)*(mpp))'''
    #get xvalues to average
    leftPolyX=leftPoly[0]*imgH**2+leftPoly[1]*imgH+leftPoly[2]
    rightPolyX=rightPoly[0]*imgH**2+rightPoly[1]*imgH+rightPoly[2]
    #take average, subtract half the img width, multiply by mpp(meters per pixel) then return result(returns neg value if to right of center pos if to left)
    return (np.mean([leftPolyX,rightPolyX])-imgW/2)*mppx

def getCurvature(imgH,leftPoly,rightPoly,mppx=(3.7/700),mppy=(1/24)):
    '''returns average radius of curvature of the 2 lane lines in pixels'''
    a=leftPoly[0]*mppx/(mppy**2)
    b=leftPoly[1]*mppx/mppy
    leftROC=((1+(2*a*imgH+b)**2)**1.5)/np.absolute(2*a)
    a=rightPoly[0]*mppx/(mppy**2)
    b=rightPoly[1]*mppx/mppy
    rightROC=((1+(2*a*imgH+b)**2)**1.5)/np.absolute(2*a)
    return np.mean([leftROC,rightROC])

def fillLane(img,leftPoly,rightPoly):#TODO: Probably a cv2 function to fill in a shape with curved sides use that
    '''fills area between polynomials in green'''
    img=np.zeros_like(img)

    for y in range(img.shape[0]):
        minX=np.int(leftPoly[0]*y**2+leftPoly[1]*y+leftPoly[2])
        maxX=np.int(rightPoly[0]*y**2+rightPoly[1]*y+rightPoly[2])
        img[y,minX:maxX]=[0,255,0]
        #for x in range(minX,maxX+1):#assumes polynomials don't give invalid x values(>imgwidth+1 or <0)
            #img[y][x]=[0,255,0]
    return img

def detectLane(img,leftPoly=None,rightPoly=None):
    '''detects lines and their polynomials,calculate cuvature and distance from center, then draw all that onto original image'''
    img=cv2.undistort(img, mtx, dist, None, mtx)
    binImg=binary(warp(img))
    ret=getPoly(binImg,leftPoly,rightPoly)

    outImg=cv2.addWeighted(img,1,warp(fillLane(binImg,ret[0],ret[1]),False),0.7,0)

    cv2.putText(outImg,'Car is '+str(distFromCenter(img.shape[0],img.shape[1],ret[0],ret[1]))+' meters from lane center',(40,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(outImg,'Lane has a radius of curvature of '+str(getCurvature(img.shape[0],ret[0],ret[1]))+' meters',(40,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    return outImg,ret[0],ret[1]

def imgDir(dir,display=True):
    '''dir is path to directory where all the images(and only the images) are stored
    images can be any format, and results are outputed same type as input
    '''
    #load filenames of all the images want to detect lines in and create output folder
    filenames=os.listdir(dir)
    if os.path.exists(dir+'_output'):
        shutil.rmtree(dir+'_output')
    os.mkdir(os.path.join(os.getcwd(),dir+'_output'))

    for img in filenames:
        absImgPath=os.path.join(os.getcwd(),dir,img)
        outImg=detectLane(cv2.imread(absImgPath))[0]#read in the image and detect lane lines
        cv2.imwrite(dir+'_output/'+img,outImg)#save image with lane lines to output directory
        if display:
            cv2.imshow(img,outImg)#show image with lane lines to screen
            while True:
                if cv2.waitKey(1) & 0xFF==ord('q'):#when they hit 'q' close the image on screen and move to next one
                    break
            cv2.destroyAllWindows()

def videoDir(dir,display=True):
    '''dir is path to directory containing multiple videos to process,
    vid can be any format(that your computer has the proper codecs for)
    this will output whatever video type vid is'''
    #load filenames of all the videos want to detect lines in and create output folder
    filenames=os.listdir(dir)
    if os.path.exists(dir+'_output'):
        shutil.rmtree(dir+'_output')
    os.mkdir(os.path.join(os.getcwd(),dir+'_output'))

    for vid in filenames:
        absVidPath=os.path.join(os.getcwd(),dir,vid)
        cap=cv2.VideoCapture(absVidPath)

        #create VideoWriter object to create video file with same metadata as vid and put it into the output directory
        absOutVidPath=os.path.join(os.getcwd(),dir+'_output',vid)
        out=cv2.VideoWriter(absOutVidPath,int(cap.get(cv2.CAP_PROP_FOURCC)),cap.get(cv2.CAP_PROP_FPS),(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        leftPoly=None
        rightPoly=None
        while cap.isOpened():#while vid still playing
            ret,frame=cap.read()
            if ret==True:
                outFrame,leftPoly,rightPoly=detectLane(frame,leftPoly,rightPoly)#detect lane lines in the frame
                out.write(outFrame)#write frame with lane lines drawn on to output video
                if display:
                    cv2.imshow(vid,outFrame)#show fram with lane lines to screen
                    if cv2.waitKey(1) & 0xFF==ord('q'):#stop processing this video when user hits 'q'
                        break
            else:
                break
        #once done close streams
        cap.release()
        out.release()
        #close window playing video
        cv2.destroyAllWindows()

'''running on all test images and videos(videos not working or at least taking really long)'''
ret, mtx, dist, rvecs, tvecs=calibrate('camera_cal',9,6,False)
imgDir('test_images')#test the function imgDir on the test_images directory
videoDir('test_videos',False)#test videoDir on the test_videos directory

'''Testing'''
'''outImg=getPoly(binary(warp(cv2.undistort(cv2.imread('test_images/test1.jpg'), mtx, dist, None, mtx))),getImg=True)[0]#read in the image and detect lane lines
#outImg=detectLane(cv2.imread('test_images/test1.jpg'))[0]
cv2.imwrite('./examples/test1_color_fit_lines.jpg',outImg)#save image with lane lines to output directory
cv2.imshow('test1.jpg',outImg)#show image with lane lines to screen
while True:
    if cv2.waitKey(1) & 0xFF==ord('q'):#when they hit 'q' close the image on screen and move to next one
        break
cv2.destroyAllWindows()'''
