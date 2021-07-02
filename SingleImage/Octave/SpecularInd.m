function I_d = SpecularInd(input)

pkg load image %

I=input;
[n_row, n_col, n_ch] = size(I);
nu = 0.5;
I = reshape(I, n_row*n_col, 3);
% Calculate specular-free image
I_min = min(I, [], 2);
T_v = mean(I_min) + nu * std(I_min);
% I_MSF = I - repmat(I_min, 1, 3) .* (I_min > T_v) + T_v * (I_min > T_v);

% Calculate specular component
beta_s = (I_min - T_v) .* (I_min > T_v) + 0;

% Estimate largest region of highlight
IHighlight = reshape(beta_s, n_row, n_col, 1);
IHighlight = mat2gray(IHighlight);
IHighlight = im2bw(IHighlight, 0.1); %#ok
IDominantRegion = bwareafilt(IHighlight, 1, 'largest');

% Dilate largest region by 5 pixels to obtain its surrounding region
se = strel('square', 5);
ISurroundingRegion = imdilate(IDominantRegion, se);
ISurroundingRegion = logical(imabsdiff(ISurroundingRegion, IDominantRegion));

% Solve least squares problem
I_dom = mean(I(IDominantRegion, :));
I_sur = mean(I(ISurroundingRegion, :));
beta_dom = mean(beta_s(IDominantRegion, :));
beta_sur = mean(beta_s(ISurroundingRegion, :));
k = (I_dom - I_sur) / (beta_dom - beta_sur);

% Estimate diffuse and specular components
I_d = reshape(I-min(k)*beta_s, n_row, n_col, n_ch);

end
