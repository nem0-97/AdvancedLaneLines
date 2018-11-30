**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: (/examples/test1_undistorted.jpg) "Undistorted"
[image2]: (/test_images/test1.jpg) "Road Transformed"
[image3]: /examples/test1_binary_combo.jpg "Binary Example"
[image4]: /examples/test1_warped.jpg "Warp Example"
[image5]: /examples/test1_color_fit_lines.jpg "Fit Visual"
[image6]: /test_images_output/test1.jpg "Output"
---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate` function, lines #13 through #43 of the file called `DetectLane.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction with the values gotten from `cv2.calibrateCamera()` on line #227 whenever I start trying to detect the lane in an image. Using `cv2.undistort()` on test_images/test1.jpg yields this result:

!["Undistorted"](/examples/test1_undistorted.jpg)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
###### Starting image:
!["Road Transformed"](/test_images/test1.jpg)
###### Undistorted:
!["Undistorted"](/examples/test1_undistorted.jpg)

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp`, which appears in lines #1 through #8 in the file `DetectLane.py` .The `warp` function takes as inputs an image (`img`), as well as if you are going to a top down view or back to start view(for when you want to overlay over original image). I started testing some points and ultimately figured out the points I used for `src` and  `dst` to work well.

###### Top down view:
!["Warp Example"](/examples/test1_warped.jpg)

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image The code for this step is contained in the 'binary' function, lines #62 through #91 of the file called `DetectLane.py`, it conducts sobelX to get the x gradients on the L channel of an HLS version of the image and then identifies pixels that meet the threshold it then also identifies pixels that meet a certain threshold along the S channel and then colors all pixels identified by either method white.
Here's an example of my output for this step.(in this image the pixels identified by sobelX and a threshold are green, and the ones from a theshold on the S channel are red, in the program they are all made white before detecting the polynomials)

###### Binary Image:
!["Binary Example"](/examples/test1_binary_combo.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A function called `getPoly`, lines #165 to #195 of `DetectLane.py`, is then called, it excepts an image, `img` and 2 arguments `leftPoly`, `rightPoly`(the coefficients for the polynomials found in the previous frame, or `None` if first frame of video or no previous image). This function will run a `slidingWindows` search to collect pixels to try and fit the polynomial to if  `leftPoly` or `rightPoly` are `None`, otherwise it will run `searchFromPrior` with `leftPoly` and `rightPoly` to collect those pixels to fit from around the polynomials passed in. From there it will come up with the coefficients for a polynomial to fit the collected pixels using `np.polyfit` it will then return these polynomials(and a visualization if you want one)(all polynomials here are 2nd degree):

!["Fit Visual"](/examples/test1_color_fit_lines.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `getCurvature`, lines #208 through #216 of `DetectLane.py`. I use the images height,`imgH` as the y value to calculate the curvature of both `leftPoly` and `rightPoly` at,(using the formula for curvature provided in the lessons), I then use the average of these 2 values as my 'actual radius of curvature'


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `detectLane` function, line #233 of  `DetectLane.py`, I call my `fillLane` function, lines #218 to #226 of `DetectLane.py`, passing it the binary image(`img`), `leftPoly`, and `rightPoly`, it then colors all points between the 2 polynomials that identify the lane lines green and everything else black and returning the resulting image. I then use `warp` this time with `top=False` so that it applies a perspective transform that reverses the one we did earlier to get a topdown view of the road so that now it is back as it was in the original undistorted image, I then use `cv2.addWeighted` to overlay this onto the undistorted image and return that.   Here is an example of my result on a test image:

!["Output"](/test_images_output/test1.jpg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
(I have since changed some things since writing this, but I think everything here still holds true, I think that a lot of things can be greatly refined and have more tuning and experimenting applied to things like threshold and src and dst points, as well as making things more efficient)

To improve the results on the challenge videos I could set it up so that if it detects the polynomials get too close or the same it stops using search from prior and does slidingWindows(I think it will stop the lane from disappearing in the first challenge video).

I could improve my thresholding or perspective transform so it either cuts out hood of car, or so that my thresholding filters out the pixels detected on the hood, it sometimes is chosen as the base for the polynomial of a lane line producing an inaccurate polynomial, especially on a side where there are dashed lines. My pipeline is inefficient I think(at least for videos), there are probably more efficient ways to do a lot of the things I do, especially coloring in every pixel green or black before transforming back and overlaying onto the original image, if I had more time to look around and learn more about numpy and openCV I think that I would be able to find more efficient ways to do a lot of the things my code does with those libraries. Another thing I did was I tried to use a lot of the same functions for both video and images, this means I have a lot more checks and variables being unnecessarily passed around, I also could probably combine some functions since a lot of them require the same variables so I am not passing them around unnecessarily.I think if the polynomial happens to give me an x value that falls off screen when I try to draw a line that fits it, it will raise an error, since I simply just try to get that pixel and make it green when drawing the line. I could also get more accurate measurements of meters per pixel to improve `distFromCenter` and `getCurvature`
