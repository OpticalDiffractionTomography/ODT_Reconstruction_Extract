clear all
close all
%% Initialize GPU
g=gpuDevice(1);
reset(g);
%% Define parameters
load('Retrieved_Field.mat');
[xx, yy, frame]=size(retPhase);
crop_size=xx;           %Retrieved field is cropped in Fourier space
f_dx2=f_dx-f_dx(49);    %subtract maxpoint
f_dy2=f_dy-f_dy(49);
original_size=xSize;    %size of original hologram
n_m=1.36275;            %RI of medium
n_s=n_m+0.04;           %maximum RI value in the colormap (only visualization)


ZP=round(1.5*xx/2)*2;	%Zeropadding the field to decrease the Fourier space resolution
crop_factor=crop_size/original_size;
res2=res/crop_factor;	% Pixel resolution in the retrieved Field

padd_factor=ZP/crop_size;
kres=1/(res*ZP)*crop_factor;

f_dx2=f_dx2*padd_factor;f_dy2=f_dy2*padd_factor;
k0_x=kres*f_dx2;        % for actual value, multiply resolution
k0_y=kres*f_dy2;

k0=1/lambda;
k0_z=real(sqrt((n_m*k0)^2-(k0_x).^2-(k0_y).^2)); % magnitude of absolute value is k0
%% Exclude abnormal fields
excludeFrame=[];
temp=mean(squeeze(mean(abs(retPhase),1)));
excludeFrame=[excludeFrame, find(abs(temp)>.5)];
temp=temp-circshift(temp,1);
excludeFrame=[excludeFrame,  find(abs(temp)>0.02)];

for kkk=1:frame
    p2=squeeze(retPhase(:,:,kkk));
    if isnan(max(max(p2)))
        excludeFrame=[excludeFrame,kkk];
    end
end
%% Fourier diffraction theorem
ZP2=512;
ZP3=256;
res3=res2*ZP/ZP2;
res4=res2*ZP/ZP3;

ORytov=gpuArray(single(zeros(ZP2,ZP2,ZP3)));    % Assign 3D Fourier space
Count=(single(zeros(ZP2,ZP2,ZP3)));             %Count is for averaging out multiple mappings at one pixel.

frameList=(1:frame);
frameList(excludeFrame)=[];

for kk=frameList
    FRytov=squeeze(log(retAmplitude(:,:,kk))+1i*retPhase(:,:,kk));
    FRytov=gpuArray(padarray(FRytov,[round((ZP-xx)/2) round((ZP-yy)/2)],'symmetric'));
    UsRytov=fftshift(fft2(FRytov)).*(res2)^2;   %2D Fourier spectra
    UsRytov=circshift(UsRytov,[round(f_dx2(kk)) round(f_dy2(kk))]);
    xr=(ZP*res2*NA/lambda);                     %Radius according to NA/lambda
    UsRytov=UsRytov.*~mk_ellipse(xr,xr,ZP,ZP);
    
    [ky kx]=meshgrid(kres*(-floor(ZP/2)+1:floor(ZP/2)),kres*(-floor(ZP/2)+1:floor(ZP/2)));
    kz=real(sqrt((n_m*k0)^2-kx.^2-ky.^2));      %Assign coordinates of the surface of Ewald sphere
    Kx=kx-k0_x(kk);Ky=ky-k0_y(kk);Kz=kz-k0_z(kk);
    Uprime=1i.*2*2*pi*kz.*UsRytov;              %Fourier diffraction theorem
    
    xind=find((kz>0).*~mk_ellipse(xr,xr,ZP,ZP)...
        .*(Kx>(kres*(-floor(ZP2/2)+1)))...
        .*(Ky>(kres*(-floor(ZP2/2)+1)))...
        .*(Kz>(kres*(-floor(ZP3/2)+1)))...
        .*(Kx<(kres*(floor(ZP2/2))))...
        .*(Ky<(kres*(floor(ZP2/2))))...
        .*(Kz<(kres*(floor(ZP3/2)))));
    
    Uprime=Uprime(xind);
    Kx=Kx(xind);
    Ky=Ky(xind);
    Kz=Kz(xind);
    
    Kx=round(Kx/kres+ZP2/2);Ky=round(Ky/kres+ZP2/2);Kz=round(Kz/kres+ZP3/2);
    Kzp=(Kz-1)*ZP2^2+(Ky-1)*ZP2+Kx;
    temp=ORytov(Kzp);
    ORytov(Kzp)=temp+Uprime;
    Count(Kzp)=Count(Kzp)+1;
end
ORytov=gather(ORytov);
ORytov(Count>0)=ORytov(Count>0)./Count(Count>0);
ORytov=gpuArray(ORytov);
clear Count UsRytov Uprime temp FRytov;
Reconimg=ifftn(ifftshift(ORytov))./(res3^2*res4);
Reconimg=n_m*sqrt(1-Reconimg.*(lambda/(n_m*2*pi))^2);
%% Iterative non-negativity constraint
ORytov_index=(find(abs(ORytov)>0));
ORytov=ORytov(ORytov_index);
normFact=1./(res3^2*res4);
for mm = 1:100
    id = (real(Reconimg)>n_m);
    Reconimg(id)=n_m-1i*imag(Reconimg(id));
    clear id;
    Reconimg=-(2*pi*n_m/lambda)^2.*(Reconimg.^2/n_m^2-1);
    ORytov_new=fftshift(fftn(Reconimg))/normFact;
    ORytov_new(ORytov_index)=ORytov;
    Reconimg=(ifftn(ifftshift(ORytov_new)))*normFact;
    Reconimg=n_m*sqrt(1-Reconimg.*(lambda/(n_m*2*pi))^2);
end
Reconimg=real(gather(fftshift(Reconimg,3)));
%% Cropping padded RI tomogram
ZP4=round(size(Reconimg,1)/3);
ZP5=round(size(Reconimg,3)/6);
Reconimg=Reconimg(end/2-ZP4+1:end/2+ZP4,end/2-ZP4+1:end/2+ZP4,end/2-ZP5+1:end/2+ZP5);
ZP4=round(size(Reconimg,1));
ZP5=round(size(Reconimg,3));
%% RI tomogram visualization
figure(101)
subplot(221),imagesc(((1:ZP4)-ZP4/2)*res3,((1:ZP4)-ZP4/2)*res3,conv2(max(real(Reconimg),[],3),fspecial('disk',0.7)),[n_m+0.005 n_s+0.01]),axis image,colorbar
subplot(224),imagesc(((1:ZP5)-ZP5/2)*res4,((1:ZP4)-ZP4/2)*res3,conv2(real(squeeze(Reconimg(:,end/2+0,:))),fspecial('disk',0.7)),[n_m+0.005 n_s+0.0]),axis image,colorbar
subplot(223),imagesc(((1:ZP4)-ZP4/2)*res3,((1:ZP4)-ZP4/2)*res3,conv2(real(squeeze(Reconimg(:,:,end/2+1))),fspecial('disk',0.7)),[n_m+0.005 n_s+0.0]),axis image,colorbar
colormap('jet')