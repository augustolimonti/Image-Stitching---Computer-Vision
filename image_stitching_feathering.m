% 
% 
hillscene = imageDatastore('balcony');
% montage(hillscene.Files);

% Read the first image from the image set.
I = readimage(hillscene,1);
% Initialize features for Image One
grayImage = im2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage,points);

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the hill images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
numImages = numel(hillscene.Files);
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages
    
    % Store points and features for I(n-1).
    I_prev = readimage(hillscene, n-1);
    pointsPrevious = points;
    featuresPrevious = features;
        
    % Read I(n).
    I = readimage(hillscene, n);
    
    % Convert image to grayscale.
    grayImage = im2gray(I);    
    
    % Save image size.
    imageSize(n,:) = size(grayImage);
    
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);
  
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
%     figure
%     showMatchedFeatures(I_prev, I, matchedPointsPrev, matchedPoints, 'montage');
%     legend('Inlier points in I1', 'Inlier points in I2');
    
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform2D(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

% 
% Compute the output limits for each transform.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end
% 
% Next, compute the average X limits for each transforms and find the image 
% that is in the center. Only the X limits are used here because the scene 
% is known to be horizontal. If another set of images are used, 
% both the X and Y limits may need to be used to find the center image.
avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

% Finally, apply the center image's inverse transform to all the others.
Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).T = tforms(i).T * Tinv.T;
end
% 
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end
% % 
maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% % Create the panorama.
% for i = 1:numImages
%     
%     I = readimage(hillscene, i);
%     
% %     Transform I into the panorama.
%     warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
%     if i == 1
%          I = readimage(hillscene, i);
%          one_warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
%     
%     elseif i == 2
%          I = readimage(hillscene, i);
%          two_warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
%     else
%         I = readimage(hillscene, i);
%         three_warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
%     end
% %     Generate a binary mask.    
%     mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
%     mask = imcomplement(mask);
%     mask(1, :) = 1;
%     mask(end, :) = 1;
%     mask(:, 1) = 1;
%     mask(:, end) = 1;
%     mask = bwdist(mask, 'euclidean');
%     mask = mask ./ max(max(mask));
%     if i == 1
%         mask_one = mask;
%     elseif i == 2
%         mask_two = mask;
%     else
%         mask_three = mask;
%         new_mask = mask_one | mask_two | mask_three;
% %         figure; imshow(new_mask);
%         new_img = (mask_one.*im2single(one_warpedImage)+mask_two.*im2single(two_warpedImage) + mask_three.*im2single(three_warpedImage))./(mask_one+mask_two+mask_three);
%         figure; imshow(new_img); 
%     end
% end
% hillscene = imageDatastore('hill');
for i = 1:numImages
    
    I = readimage(hillscene, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    figure;
    imshow(panorama);
end

figure;
imshow(panorama);