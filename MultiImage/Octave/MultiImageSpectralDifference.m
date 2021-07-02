function I_d = MultiImageSpectralDifference(I,I1,I2)


pkg load image %

[n_row, n_col, n_ch] = size(I);

nu = 0.5;

I = reshape(I, n_row*n_col, 3);

I_min = min(I, [], 2);
T_v = mean(I_min) + nu * std(I_min);


beta_s = (I_min - T_v) .* (I_min > T_v) + 0;

IHighlight = reshape(beta_s, n_row, n_col, 1);
IHighlight = mat2gray(IHighlight);
IHighlight = im2bw(IHighlight, 0.1); %#ok
IDominantRegion = bwareafilt(IHighlight, 1, 'largest');


se = strel('square', 5);
ISurroundingRegion = imdilate(IDominantRegion, se);
ISurroundingRegion = logical(imabsdiff(ISurroundingRegion, IDominantRegion));


I_dom = mean(I(IDominantRegion, :));
I_sur = mean(I(ISurroundingRegion, :));
beta_dom = mean(beta_s(IDominantRegion, :));
beta_sur = mean(beta_s(ISurroundingRegion, :));
k = (I_dom - I_sur) / (beta_dom - beta_sur);


I_d = reshape(I-min(k)*beta_s, n_row, n_col, n_ch);

end