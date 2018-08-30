%This code demonstrates the techniques of color image registration
%Code written by Sai Saradha K.L. (MS, Computer Engineering, Fall 2016)
%Getting the image set:
clc;
clear;
%Read-in the image :
full_img=imread('E:\GraduateLife\Fall_2016_Semester I\EECS490_DIP\Assignments\Project #5\Images\00895u.tif');
figure, imshow(full_img);
 
%Height of the image:
height=floor(length(full_img)/3);
 
%Color channel separation:
blue_img=full_img(1:height, :);
green_img=full_img(height+1:2*height, :);
red_img=full_img(2*height+1:end, :);
figure, imshow(blue_img);
figure, imshow(green_img);
figure, imshow(red_img);
 
%Combine the three channels before registration :
comb_orig=cat(3, red_img, green_img, blue_img);
figure, imshow(comb_orig);
 
%%
%MATLAB SURF Function :
%SURF for blue and red :
tic
ptsOriginal  = detectSURFFeatures(blue_img);
ptsDistorted = detectSURFFeatures(red_img);
 
[featuresOriginal,   validPtsOriginal]  = extractFeatures(blue_img,  ptsOriginal);
[featuresDistorted, validPtsDistorted]  = extractFeatures(red_img, ptsDistorted);
 
indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));
figure;
showMatchedFeatures(blue_img,red_img,matchedOriginal,matchedDistorted);
title('Putatively matched points (including outliers)');
 
[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');
figure;
showMatchedFeatures(blue_img,red_img, inlierOriginal, inlierDistorted);
title('Matching points (inliers only)');
legend('ptsOriginal','ptsDistorted');
 
Tinv  = tform.invert.T;
 
ss = Tinv(2,1);
sc = Tinv(1,1);
scale_recovered = sqrt(ss*ss + sc*sc)
theta_recovered = atan2(ss,sc)*180/pi
 
outputView = imref2d(size(blue_img));
recovered  = imwarp(red_img,tform,'OutputView',outputView);
 
figure, imshowpair(blue_img,recovered,'blend');
 
%SURF for blue and green:
 
ptsOriginal  = detectSURFFeatures(blue_img);
ptsDistorted = detectSURFFeatures(green_img);
 
[featuresOriginal,   validPtsOriginal]  = extractFeatures(blue_img,  ptsOriginal);
[featuresDistorted, validPtsDistorted]  = extractFeatures(green_img, ptsDistorted);
 
indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));
figure;
showMatchedFeatures(blue_img,green_img,matchedOriginal,matchedDistorted);
title('Putatively matched points (including outliers)');
 
[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');
figure;
showMatchedFeatures(blue_img,green_img, inlierOriginal, inlierDistorted);
title('Matching points (inliers only)');
legend('ptsOriginal','ptsDistorted');
 
Tinv  = tform.invert.T;
 
ss = Tinv(2,1);
sc = Tinv(1,1);
scale_recovered = sqrt(ss*ss + sc*sc)
theta_recovered = atan2(ss,sc)*180/pi
 
outputView = imref2d(size(blue_img));
recovered2  = imwarp(green_img,tform,'OutputView',outputView);
 
figure, imshowpair(blue_img,recovered2,'blend');
 
sift_final=cat(3, recovered, recovered2, blue_img);
figure, imshow(sift_final);
toc
 
%%
%Automatic cropping final:
%Thresholding on gradient image: Use Otsu's thresholding method
img=rgb2gray(sift_final);
%histogram of the image
histo=imhist(img);
L=numel(histo);
%Normalized histogram
norm_histo=histo/sum(histo);
%calculate threshold value k
sum_t1=zeros(L,1);
mean_t1=zeros(L,1);
mg=zeros(L,1);
 
%Compute cumulative sum (sum_t1) and average cumulative mean(mean_t1)
for i=1:L
    if(i==1)
        sum_t1(1,1)=norm_histo(1,1);
        mean_t1(1)=0;
        mg(1)=0;
    end
    if(i>1)
    sum_t1(i,1)=sum_t1(i-1,1)+norm_histo(i,1);
    mean_t1(i,1)=mean_t1(i-1,1)+((i-1)*norm_histo(i,1));
    mg(i)=mg(i-1)+((i-1)*norm_histo(i,1));
    end
end
%Global intensity mean (0,1,2,...L-1)
mgf=mg(end);
%between class variance
sigma_bsqrd=(mgf * sum_t1 - mg).^2 ./ (sum_t1 .* (1 - sum_t1));
max_sigmabsqrd = max(sigma_bsqrd);
isfinite_maxval = isfinite(max_sigmabsqrd);
if isfinite_maxval
    thres_avg = mean(find(sigma_bsqrd == max_sigmabsqrd));
% Normalize the threshold to the range [0, 1].
    t = (thres_avg - 1) / (L - 1);
else
    t = 0.0;
end
thresh_img=im2bw(img,0.51*t);
figure, imshow(thresh_img);
figure, imshowpair(img, thresh_img,'montage');
% thresh_alnum=img < t;
% figure, imshow(thresh_alnum);
% figure, imshowpair(img,thresh_alnum,'montage');
x=zeros;
r_size=ceil((1/20)*size(thresh_img,1));
c_size=ceil((1/20)*size(thresh_img,2));
 
for i=1:r_size
    x(i)=mean(thresh_img(i,:));
end
a=find(x<=0.4);
toprow=max(a);
if isempty(toprow)
    toprow=1;
end
x=zeros;
for i=1:c_size
    x(i)=mean(thresh_img(:,i));
end
b=find(x<=0.4);
leftcolumn=max(b);
if isempty(leftcolumn)
    leftcolumn=1;
end
 
x=zeros;
for i=1:r_size
    x(i)=mean(thresh_img(end-i-1,:));
end
c=find(x<=0.4);
bottomrow=size(thresh_img,1)-min(c);
if isempty(bottomrow)
    bottomrow=size(thresh_img,1);
end
x=zeros;
for i=1:c_size
    x(i)=mean(thresh_img(:,end-i-1));
end
d=find(x<=0.4);
rightcolumn=size(thresh_img,1)-min(d);
if isempty(rightcolumn)
    rightcolumn=size(thresh_img,2);
end
 
%Final image:
final_cropped=sift_final(toprow:bottomrow,leftcolumn:rightcolumn,:);
figure, imshow(final_cropped);
 
 
%%
%Image enhancement:
im1=histeq(final_cropped(:,:,1));
im2=histeq(final_cropped(:,:,2));
im3=histeq(final_cropped(:,:,3));
 
 
histeq_img=cat(3,im1,im2,im3);
figure, imshow(histeq_img);

function [red_shift_new, green_shift_new] = translation(samp_img, red_shift, green_shift, search, down)
    %Get the size of the image to check if it is too large :
    [i_h, i_w ~] = size(samp_img);
    %We are going to reduce the image to 1024 x 1024 or slightly lesser than that;
    h_offset=1;
    w_offset=1;
    end_h=i_h;
    end_w=i_w;
    if (i_w > 1024)
        h_offset=(i_h/2-512)+round(search/down);
        w_offset=(i_w/2-512)+round(search/down);
        
    end
 
    %Now use a sobel edge image to find the maximum correlation :
    %smoothing the image with average filter:
        h=ones(5,5)/25;
        smoothed_img=imfilter(samp_img,h);
        figure, imshow(smoothed_img);
 
        %Use Canny edge detector :
        %Red edge image:
        x_red=h_offset+red_shift(1,1);
        y_red=w_offset+red_shift(1,2);
        if(i_w > 1024)
            end_h=x_red+512;
            end_w=y_red+512;
        end
        x_red_range=x_red:end_h;
        y_red_range=y_red:end_w;
        red_canny=double(edge(smoothed_img(x_red_range,y_red_range,1),'canny',[0.05 0.19],3.5));
        figure, imshow(red_canny);
        
        %Green edge image:
        x_green=h_offset+green_shift(1,1);
        y_green=w_offset+green_shift(1,2);
        if(i_w > 1024)
            end_h=x_red+512;
            end_w=y_red+512;
        end
        x_green_range=x_green:end_h;
        y_green_range=y_green:end_w;
        green_canny=double(edge(smoothed_img(x_green_range,y_green_range,2),'canny',[0.05 0.19],3.5));
        figure, imshow(green_canny);
        
        %Now concatenate the red and blue channels :
        comb_edge=cat(3,red_canny, green_canny);
        
        %Blue edge image:
        if(i_w > 10024)
            end_h=h_offset+512;
            end_w=h_offset+512;
        end
        x_blue_range=(h_offset+search):(end_h-search);
        y_blue_range=(w_offset+search):(end_w-search);
        blue_canny=double(edge(smoothed_img(round(x_blue_range),round(y_blue_range),3),'canny',[0.05 0.19],3.5));
        figure, imshow(blue_canny);
        
        %Now, let us find the maximum cross correlation :
        normcc=zeros;
        x_sh=zeros;
        y_sh=zeros;
        
        for i=1:2
            normmax=0;
            for j=-search:1:search
                for k = -search:1:search
                  current_window=comb_edge(j+1+search:j+end-search,k+1+search:k+end-search,i)  ;
                  normcc(j+1+search, k+1+search, i)=sum(current_window(:).*blue_canny(:))/(norm(current_window(:))*norm(blue_canny(:)));
                  if normcc(j+1+search, k+1+search, i) > normmax
                      x_sh(1,i)=j;
                      y_sh(1,i)=k;
                      normmax=normcc(j+1+search, k+1+search, i);
                  end
                end
            end
            
        end
        red_shift_new = [x_sh(1,1) y_sh(1,1)];
        green_shift_new = [x_sh(1,2) y_sh(1,2)];
        
end


   full_img=imread('E:\GraduateLife\Fall_2016_Semester I\EECS490_DIP\Assignments\Project #5\Images\01137u.tif');
    figure, imshow(full_img);
    original=full_img;
    %Cut the image into three pieces
    interval = floor(length(original(:,1))/3);
    B = single(double(original(1:interval,:))./255);
    G = single(double(original(interval+1:interval*2,:))./255);
    R = single(double(original(interval*2+1:end-mod(length(original(:,1)),3),:))./255);
 
    %Crop out the black borders
    crop = round(0.05*size(original,2));
    R = R(crop:end-crop,crop:end-crop);
    G = G(crop:end-crop,crop:end-crop);
    B = B(crop:end-crop,crop:end-crop);
 
    downR = R;
    downG = G;
    downB = B;
    num_down = 1;
 
    R_tmp = R;
    G_tmp = G;
    B_tmp = B;
tic
    [A1, T1] = vlsift_match(B_tmp,G_tmp,1000, 1);
    [A2, T2] = vlsift_match(R_tmp,G_tmp,1000, 1); 
 
 
        %For B to G
        boxB = [1  size(B,2) size(B,2)  1 ;
                1  1           size(B,1)  size(B,1)];
        boxB_ = A1 * boxB + T1*ones(1,4)*2^num_down; %Transformed corners of B
 
        boxR = [1  size(R,2) size(R,2)  1 ;
                1  1           size(R,1)  size(R,1)];
        boxR_ = A2 * boxR + T2*ones(1,4)*2^num_down; %Transformed corners of R
 
        %Define the x and y entries for the meshgrid based on a grid that
        %can encapsulate R,G, and B
        ur = min([1 boxB_(1,:) boxR_(1,:)]):max([size(G,2) boxB_(1,:) boxR_(1,:)]);
        vr = min([1 boxB_(2,:) boxR_(2,:)]):max([size(G,1) boxB_(2,:) boxR_(2,:)]);
 
        [u,v] = meshgrid(ur,vr);
        G_ = vl_imwbackward(im2double(G),u,v); %Find G in new grid
 
        %Find the original x,y coordinates in the original image that map
        %to the coordinates in the full grid.
        H1 = inv(A1);
        uB_ = (H1(1,1) * u + H1(1,2) * v + T1(1));
        vB_ = (H1(2,1) * u + H1(2,2) * v + T1(2));
        B_ = vl_imwbackward(im2double(B),uB_,vB_) ;
 
 
        H2 = inv(A2);
        uR_ = (H2(1,1) * u + H2(1,2) * v + T2(1));
        vR_ = (H2(2,1) * u + H2(2,2) * v + T2(2));
        R_ = vl_imwbackward(im2double(R),uR_,vR_) ;
 
        %Make the final image look good and not have NANs floating around
        B_(isnan(B_)) = 0 ;
        R_(isnan(R_)) = 0 ;
        G_(isnan(G_)) = 0;
        mosaic = zeros(size(G_,1),size(G_,2),3);
        mosaic(:,:,1) = R_;
        mosaic(:,:,2) = G_;
        mosaic(:,:,3) = B_;
 
        figure();
        imshow(mosaic);
        title('Original');
    toc

function [ A, T] = vlsift_match( image1, image2, iter, thr )
 
[f1, d1] = vl_sift(image1);
[f2, d2] = vl_sift(image2);
 
%x, y coordinates for match
ac1 = f1(1:2,:);
ac2 = f2(1:2,:);
 
rows1 = (f1(1,:) < 0.97*size(image1,2)).*(f1(1,:) > 0.03*size(image1,2));
columns1 = (f1(2,:) < 0.97*size(image1,1)).*(f1(2,:) > 0.03*size(image1,1));
find1 = find(rows1.*columns1);
co1 = ac1(:,find1);
 
rows2 = (f2(1,:) < 0.97*size(image2,2)).*(f2(1,:) > 0.03*size(image2,2));
columns2 = (f2(2,:) < 0.97*size(image2,1)).*(f2(2,:) > 0.03*size(image2,1));
find2 = find(rows2.*columns2);
co2 = ac2(:,find2);
 
d1 = d1(:,find1);
d2 = d2(:,find2);
 
%Find matches
[matches, scores] = vl_ubcmatch(d1, d2,1.75); 
 
 
cor1 = co1(:, matches(1,:));
cor2 = co2(:, matches(2,:));
 
data = cell(iter, 2); 
in_num = zeros(iter, 1); 
 
 
for k = 1:iter
    index1 = matches(:, max(1,round(rand * length(matches))));
    index2 = matches(:, max(1,round(rand * length(matches))));
    
    %image coordinates
    co11 = co1(:, index1(1));
    co12 = co2(:, index1(2));
    
    %image coordinates
    co21 = co1(:, index2(1));
    co22 = co2(:, index2(2));
    
    
    
    Y = [co12, co22];
    X = [co11, co21];
    
    A_tmp = Y*pinv(X);
    T_tmp = (co22 + co12)./2 - A_tmp*(co21 + co11)./2;
    
    error = zeros(1,length(cor1));
    for i = 1:length(cor1)
        error(1,i) = norm(cor2(:,i) - A_tmp*cor1(:,i) - T_tmp);
    end
    
    inliers = find(error < thr);
    numInliers = length(inliers);
    
    data{k, 1} = A_tmp;
    data{k, 2} = T_tmp;
    in_num(k) = numInliers;
   
end
 
largest_innum = find(in_num == max(in_num));
A = data{largest_innum, 1};
T = data{largest_innum, 2};


% % [bar_conc norm_bar_conc]=hist_conc(final_cropped);
im1=histeq(final_cropped(:,:,1));
im2=histeq(final_cropped(:,:,2));
im3=histeq(final_cropped(:,:,3));
 
% im1=histeq(sift_final(:,:,1));
% im2=histeq(sift_final(:,:,2));
% im3=histeq(sift_final(:,:,3));
 
histeq_img=cat(3,im1,im2,im3);
figure, imshow(histeq_img);
 
[bar_conc norm_bar_conc]=hist_conc(histeq_img);
 
 
figure, bar(bar_conc,'Barwidth',1);
figure, bar(norm_bar_conc,'Barwidth',1);
 
%Annotation Routine
 
line([1 32],[max(norm_bar_conc)/2 max(norm_bar_conc)/2]);hold on;
 
line([33 64],[max(norm_bar_conc)/2 max(norm_bar_conc)/2]);hold on;
 
line([65 96],[max(norm_bar_conc)/2 max(norm_bar_conc)/2]);
 
text(1,max(norm_bar_conc)/2,'\mid');
 
text(32,max(norm_bar_conc)/2,'\mid');
 
text(15,max(norm_bar_conc)/1.9,'Red');
 
text(15,max(norm_bar_conc)/2.2,'32 bins');
 
text(0,max(norm_bar_conc)/1.9,'0');
 
text(28,max(norm_bar_conc)/1.9,'255');
 
text(33,max(norm_bar_conc)/2,'\mid');
 
text(64,max(norm_bar_conc)/2,'\mid');
 
text(45,max(norm_bar_conc)/1.9,'Green');
 
text(45,max(norm_bar_conc)/2.2,'32 bins');
 
text(33,max(norm_bar_conc)/1.9,'0');
 
text(60,max(norm_bar_conc)/1.9,'255');
 
text(65,max(norm_bar_conc)/2,'\mid');
 
text(96,max(norm_bar_conc)/2,'\mid');
 
text(75,max(norm_bar_conc)/1.9,'Blue');
 
text(75,max(norm_bar_conc)/2.2,'32 bins');
 
text(65,max(norm_bar_conc)/1.9,'0');
 
text(96,max(norm_bar_conc)/1.9,'255');

