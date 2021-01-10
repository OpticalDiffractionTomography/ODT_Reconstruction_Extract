function H = mk_ellipse(XR,YR,X,Y)
 
[XX YY]=meshgrid(1:X,1:Y);
H=(((XX-X/2)./XR).^2+((YY-Y/2)./YR).^2)>1.0;
return;