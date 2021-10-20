% prototype program for SIM reconstruction using inverse matrix based phase estimaton algorithm
clear all
close all
a_num=1;% number of pattern orientations
p_num=3;% phase shift times for each pattern orientation
NA = 1.2;
psize = 82;
lambda = 590;
wiener_factor=0.05;
mask_factor=0.3;
%% visualization option
show_initial_result_flag=0;
show_corrected_result_flag=0;
search_range=0.6;



path = 'D:\User\Edward\OneDrive - University Of Cambridge\OS-SIM test data\Meng mito 2.tif';

% for imFile = 1:length(files) 
% %filepath=fullfile(path,files(imFile));%replace with your file's path
% if isa(files,'char')
%     filepath = strcat(path,files);
% else
%     filepath = strcat(path,files{imFile});
% end
% 
% end

PSF_edge = fspecial('gaussian',5,40);

for p=1:p_num
        temp = double(imread(path,p));
        temp = temp-min(temp(:)); temp = temp./max(temp(:));
        noiseimage(:,:,1,p) = temp;
end
[xsize, ysize] = size(noiseimage(:,:,1));
[Y,X]=meshgrid(1:ysize,1:xsize);

xc=floor(xsize/2+1);% the x-coordinate of the center
yc=floor(ysize/2+1);% the y-coordinate of the center
yr=Y-yc;
xr=X-xc;
R=sqrt((xr).^2+(yr).^2);% distance between the point (x,y) and center (xc,yc)
%% Generate the PSF
pixelnum=xsize;
rpixel=NA*pixelnum*psize/lambda;
cutoff=round(2*rpixel);% cutoff frequency
ctfde=ones(pixelnum,pixelnum).*(R<=rpixel);
ctfdeSignificantPix=numel(find(abs(ctfde)>eps(class(ctfde))));
ifftscalede=numel(ctfde)/ctfdeSignificantPix;
apsfde=fftshift(ifft2(ifftshift(ctfde)));
ipsfde=ifftscalede*abs(apsfde).^2;
OTFde=real(fftshift(fft2(ifftshift(ipsfde))));
clear apsfde ctfde temp X Y
%% filter/deconvolution before using noiseimage
widefield=sum(sum(noiseimage,4),3);
widefield=quasi_wnr(OTFde,widefield,wiener_factor^2);
widefield=widefield.*(widefield>0);


widefield = mean(noiseimage,3);

for p=1:p_num
        
        noiseimage(:,:,1,p)= edgetaper(noiseimage(:,:,1,p),PSF_edge);
        noiseimage(:,:,1,p)=quasi_wnr(OTFde,squeeze(noiseimage(:,:,1,p)),wiener_factor^2);
        
        %noiseimage(:,:,ii,jj)=deconvlucy(noiseimage(:,:,ii,jj),ipsfde,3);
        %pre-deconvolution. It can be applied to suppress noises in experiments
        noiseimage(:,:,1,p)=noiseimage(:,:,1,p).*(noiseimage(:,:,1,p)>0);

end
widefield=widefield./max(widefield(:))*max(noiseimage(:));



separated_FT=zeros(xsize,ysize,a_num,3);
noiseimagef=zeros(size(noiseimage));
for ii=1:a_num
    re0_temp=zeros(xsize,ysize);
    rep_temp=zeros(xsize,ysize);
    rem_temp=zeros(xsize,ysize);
    modulation_matrix=[1,1/2*exp(-1i*(pi*0)),1/2*exp(1i*(pi*0));...
                       1,1/2*exp(-1i*(pi*2/3)),1/2*exp(1i*(pi*2/3));...
                       1,1/2*exp(-1i*(pi*4/3)),1/2*exp(1i*(pi*4/3))];
    matrix_inv=inv(modulation_matrix);

    for jj=1:p_num
        noiseimagef(:,:,ii,jj)=fftshift(fft2(noiseimage(:,:,ii,jj)));
        re0_temp=matrix_inv(1,jj)*noiseimagef(:,:,ii,jj)+re0_temp;
        rep_temp=matrix_inv(2,jj)*noiseimagef(:,:,ii,jj)+rep_temp;
        rem_temp=matrix_inv(3,jj)*noiseimagef(:,:,ii,jj)+rem_temp;
    end

    separated_FT(:,:,ii,1)=re0_temp;
    separated_FT(:,:,ii,2)=rep_temp;
    separated_FT(:,:,ii,3)=rem_temp;
end
clear re0_temp rep_temp rem_temp noiseimage

fmask=double(sqrt(xr.^2+yr.^2)>cutoff*mask_factor);
[shiftvalue,~]=frequency_est_tirf_v2(separated_FT,0.008,fmask,show_initial_result_flag,mask_factor*cutoff);
clear separated_FT


for ii=1:a_num
    shiftvalue(ii,2,:)=shiftvalue(ii,2,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,3,:)=shiftvalue(ii,3,:)-shiftvalue(ii,1,:);
    shiftvalue(ii,1,1)=0;
    shiftvalue(ii,1,2)=0;
end

%% phase correction with inverse matrix based algorithm


%obtain a more precise estimation of the period and the directon of sinusodial pattern
[ precise_shift,~] = precise_frequency_tirf(noiseimagef,shiftvalue,search_range);

[inv_phase] = separation_matrix_correction_v3(noiseimagef,precise_shift,OTFde);

%% auto-correlation based algorithm
auto_phase=zeros(a_num,p_num);

for ii=1:a_num
    for jj=1:p_num
        f_temp=exact_shift(noiseimagef(:,:,ii,jj),...
        [-precise_shift(ii,2,1),-precise_shift(ii,2,2)],1);
     
        auto_phase(ii,jj)=angle(sum(sum(conj(noiseimagef(:,:,ii,jj)).*f_temp)));
    end
end








