import numpy as np
import cv2 as cv
import time

# If needed, comment out the section you are not using.
# Load image or images into img, img2, img_color and img_color2 variables

img_color=  cv.imread('a.jpg')
img_color = cv.resize(img_color, (0, 0), fx = 0.1, fy = 0.1)
img = cv.imread('a.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)

#FAST
fast = cv.FastFeatureDetector_create()
start_time = time.time()
kp_FAST = fast.detect (img, None)
detect_time = time.time() - start_time
print(f"Time for FAST to detect keypoints: {detect_time:.4f} seconds")
img2a = cv.drawKeypoints(img_color, kp_FAST, None, color = (0,255,0))
print ('FAST') 
print(len(kp_FAST))

#GFTT
start_time = time.time()
gftt = cv.GFTTDetector_create(maxCorners=500, qualityLevel=0.01, minDistance=3)
detect_time = time.time() - start_time
kp_GFTT = gftt.detect(img,None)
img3 = cv.drawKeypoints(img_color, kp_GFTT, None, color = (0,255,0))
print ('GFTT') 
print (len (kp_GFTT))
print(f"Time for GFTT to detect keypoints: {detect_time:.4f} seconds")

# SIFT
sift = cv.SIFT_create()
start_time = time.time()
kp_SIFT = sift.detect(img,None)
detect_time = time.time() - start_time
img4 = img_color.copy()
img5=cv.drawKeypoints(img4,kp_SIFT,img4, color=(0, 255, 0))
print ('SIFT') 
print (len (kp_SIFT))
print(f"Time for SIFT to detect keypoints: {detect_time:.4f} seconds")

print ("Number of detected keypoints on the 1st image using FAST / GFTT / SIFT:")
print (len(kp_FAST), len(kp_GFTT), len (kp_SIFT))

#BRIEF
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp_FAST, des_FAST = brief.compute(img, kp_FAST)
kp_SIFT, des_SIFT = brief.compute(img, kp_SIFT)
kp_GFTT, des_GFTT = brief.compute(img, kp_GFTT)

############# IMG 2
img_color2=  cv.imread('b.jpg')
img_color2 = cv.resize(img_color2, (0, 0), fx = 0.1, fy = 0.1)
img2 = cv.imread('b.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.resize(img2, (0, 0), fx = 0.1, fy = 0.1)

#FAST
fast = cv.FastFeatureDetector_create()
start_time = time.time()
kp_FAST2 = fast.detect (img2, None)
detect_time = time.time() - start_time
print(f"Time for FAST to detect keypoints: {detect_time:.4f} seconds")
img22 = cv.drawKeypoints(img_color2, kp_FAST2, None, color = (0,255,0))
print ('FAST') 
print(len(kp_FAST2))

#GFTT
start_time = time.time()
gftt = cv.GFTTDetector_create(maxCorners=500, qualityLevel=0.01, minDistance=3)
detect_time = time.time() - start_time
kp_GFTT2 = gftt.detect(img2,None)
img32 = cv.drawKeypoints(img_color2, kp_GFTT2, None, color = (0,255,0))
print ('GFTT') 
print (len (kp_GFTT2))
print(f"Time for GFTT to detect keypoints: {detect_time:.4f} seconds")

# SIFT
sift = cv.SIFT_create()
start_time = time.time()
kp_SIFT2 = sift.detect(img2, None)
detect_time = time.time() - start_time
img42 = img_color2.copy()
img52=cv.drawKeypoints(img42,kp_SIFT2,img42, color=(0, 255, 0))
print ('SIFT') 
print (len (kp_SIFT2))
print(f"Time for SIFT to detect keypoints: {detect_time:.4f} seconds")


print ("Number of detected keypoints on the 2nd image using FAST / GFTT / SIFT:")
print (len(kp_FAST2), len(kp_GFTT2), len (kp_SIFT2))

#BRIEF
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
start_time = time.perf_counter()
kp_FAST2, des_FAST2 = brief.compute(img2, kp_FAST2)
detect_time = time.perf_counter() - start_time
print(f"Time for BRIEF to compute descriptors for FAST keypoints: {detect_time:.5f} seconds")

start_time = time.perf_counter()
kp_SIFT2, des_SIFT2 = brief.compute(img2, kp_SIFT2)
detect_time = time.perf_counter() - start_time
print(f"Time for BRIEF to compute descriptors for SIFT keypoints: {detect_time:.5f} seconds")

start_time = time.perf_counter()
kp_GFTT2, des_GFTT2 = brief.compute(img2, kp_GFTT2)
detect_time = time.perf_counter() - start_time
print(f"Time for BRIEF to compute descriptors for FAST keypoints: {detect_time:.5f} seconds")


cv.imshow('FAST Keypoints', img2a)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window

cv.imshow('GFTT Keypoints', img3)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window

cv.imshow('SIFT Keypoints', img5)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window

cv.imshow('FAST Keypoints', img22)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window

cv.imshow('GFTT Keypoints', img32)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window

cv.imshow('SIFT Keypoints', img52)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window

####### MATCH BRIEF
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
start_time = time.time()
matches_FAST = bf.match(des_FAST, des_FAST2)
matches_FAST = sorted(matches_FAST, key=lambda x: x.distance)
compute_time = time.time() - start_time
print(f"Time to match FAST keypoints with BRIEF descriptors: {compute_time:.5f} seconds")

start_time = time.time()
matches_GFTT = bf.match(des_GFTT, des_GFTT2)
matches_GFTT = sorted(matches_GFTT, key=lambda x: x.distance)
compute_time = time.time() - start_time
print(f"Time to match GFTT keypoints with BRIEF descriptors: {compute_time:.5f} seconds")

start_time = time.time()
matches_SIFT = bf.match(des_SIFT, des_SIFT2)
matches_SIFT = sorted(matches_SIFT, key=lambda x: x.distance)
compute_time = time.time() - start_time
print(f"Time to match SIFT keypoints with BRIEF descriptors: {compute_time:.5f} seconds")

max_matches_to_draw = 15
num_matches_FAST = min(len(matches_FAST), max_matches_to_draw)
num_matches_SIFT = min(len(matches_SIFT), max_matches_to_draw)
num_matches_GFTT = min(len(matches_GFTT), max_matches_to_draw)

# Draw matches for visualization
img_FAST_matches = cv.drawMatches(img_color, kp_FAST, img_color2, kp_FAST2, matches_FAST[:num_matches_FAST], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_SIFT_matches = cv.drawMatches(img_color, kp_SIFT, img_color2, kp_SIFT2, matches_SIFT[:num_matches_SIFT], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_GFTT_matches = cv.drawMatches(img_color, kp_GFTT, img_color2, kp_GFTT2, matches_GFTT[:num_matches_GFTT], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('FAST Matches', img_FAST_matches)
cv.imwrite('FASTMatches.jpg', img_FAST_matches)
cv.imshow('SIFT Matches', img_SIFT_matches)
cv.imwrite('SIFTMatches.jpg', img_SIFT_matches)
cv.imshow('GFTT Matches', img_GFTT_matches)
cv.imwrite('GFTTMatches.jpg', img_GFTT_matches)
cv.waitKey(0)
cv.destroyAllWindows()

#FREAK
freak = cv.xfeatures2d.FREAK_create()
# Compute FREAK descriptors 
kp_FAST, des_FAST_FREAK = freak.compute(img, kp_FAST)
kp_SIFT, des_SIFT_FREAK = freak.compute(img, kp_SIFT)
kp_GFTT, des_GFTT_FREAK = freak.compute(img, kp_GFTT)

start_time = time.perf_counter()
kp_FAST2, des_FAST_FREAK2 = freak.compute(img2, kp_FAST2)
detect_time = time.perf_counter() - start_time
print(f"Time for FREAK to compute descriptors for FAST keypoints: {detect_time:.5f} seconds")

start_time = time.perf_counter()
kp_SIFT2, des_SIFT_FREAK2 = freak.compute(img2, kp_SIFT2)
detect_time = time.perf_counter() - start_time
print(f"Time for FREAK to compute descriptors for SIFT keypoints: {detect_time:.5f} seconds")
start_time = time.perf_counter()
kp_GFTT2, des_GFTT_FREAK2 = freak.compute(img2, kp_GFTT2)
detect_time = time.perf_counter() - start_time
print(f"Time for FREAK to compute descriptors for GFFT keypoints: {detect_time:.5f} seconds")

bf = cv.BFMatcher()

start_time = time.time()
matches_FAST_FREAK = bf.match(des_FAST_FREAK, des_FAST_FREAK2)
matches_FAST_FREAK = sorted(matches_FAST_FREAK, key=lambda x: x.distance)
compute_time = time.time() - start_time
print(f"Time to match FAST keypoints with FREAK descriptors: {compute_time:.5f} seconds")

start_time = time.time()
matches_GFTT_FREAK = bf.match(des_GFTT_FREAK, des_GFTT_FREAK2)
matches_GFTT_FREAK = sorted(matches_GFTT_FREAK, key=lambda x: x.distance)
compute_time = time.time() - start_time
print(f"Time to match GFTT keypoints with FREAK descriptors: {compute_time:.5f} seconds")

start_time = time.time()
matches_SIFT_FREAK = bf.match(des_SIFT_FREAK, des_SIFT_FREAK2)
matches_SIFT_FREAK = sorted(matches_SIFT_FREAK, key=lambda x: x.distance)
compute_time = time.time() - start_time
print(f"Time to match SIFT keypoints with FREAK descriptors: {compute_time:.5f} seconds")


num_matches_FAST_FREAK = min(len(matches_FAST_FREAK), max_matches_to_draw)
num_matches_SIFT_FREAK = min(len(matches_SIFT_FREAK), max_matches_to_draw)
num_matches_GFTT_FREAK = min(len(matches_GFTT_FREAK), max_matches_to_draw)


# Draw matches for visualization
img_FAST_matches2 = cv.drawMatches(img_color, kp_FAST, img_color2, kp_FAST2, matches_FAST[:num_matches_FAST_FREAK], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_SIFT_matches2 = cv.drawMatches(img_color, kp_SIFT, img_color2, kp_SIFT2, matches_SIFT[:num_matches_SIFT_FREAK], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_GFTT_matches2 = cv.drawMatches(img_color, kp_GFTT, img_color2, kp_GFTT2, matches_GFTT[:num_matches_GFTT_FREAK], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('FAST Matches (FREAK)', img_FAST_matches)
cv.imwrite('FASTMatches(FREAK).jpg', img_FAST_matches2)
cv.imshow('SIFT Matches (FREAK)', img_SIFT_matches2)
cv.imwrite('SIFTMatches(FREAK)..jpg', img_SIFT_matches2)
cv.imshow('GFTTMatches(FREAK)', img_GFTT_matches2)
cv.imwrite('GFTTMatches(FREAK).jpg', img_GFTT_matches2)
cv.waitKey(0)
cv.destroyAllWindows()


