function ImageDispMulti(XImpS,XImpO,XImpL,XImpG, XRaw)
    lTest = length(XImpS)/4;
    XImpS(XImpS<-1) = 0;
    XImpO(XImpO<-1) = 0;
    XImpL(XImpL<-1) = 0;
    XImpG(XImpG<-1) = 0; 
    XImp = [XImpS(1:lTest);XImpO(1:lTest);XImpL(1:lTest);XImpG(1:lTest);XImpS(lTest+1:2*lTest);XImpO(lTest+1:2*lTest);XImpL(lTest+1:2*lTest);XImpG(lTest+1:2*lTest);XImpS(2*lTest+1:3*lTest);XImpO(2*lTest+1:3*lTest);XImpL(2*lTest+1:3*lTest);XImpG(2*lTest+1:3*lTest);XImpS(3*lTest+1:4*lTest);XImpO(3*lTest+1:4*lTest);XImpL(3*lTest+1:4*lTest);XImpG(3*lTest+1:4*lTest)];

    XImp = flipud(XImp);
    XRaw = flipud(XRaw);
    scaledX = rescale(XRaw,0.1,0.9,'InputMin',min(XRaw,[],2),'InputMax',max(XRaw,[],2));
    Xrgb = [];
    Factor=1;
    Xrgb(:,:,1) = broadenRows(scaledX,Factor);
    Xrgb(:,:,2) = broadenRows(scaledX,Factor);
    Xrgb(:,:,3) = broadenRows(scaledX,Factor);
    XrgbL = cat(1,Xrgb(1,:,:),Xrgb(1,:,:),Xrgb(1,:,:),Xrgb(1,:,:),Xrgb(2,:,:),Xrgb(2,:,:),Xrgb(2,:,:),Xrgb(2,:,:),Xrgb(3,:,:),Xrgb(3,:,:),Xrgb(3,:,:),Xrgb(3,:,:),Xrgb(4,:,:),Xrgb(4,:,:),Xrgb(4,:,:),Xrgb(4,:,:));
    figure()
    hold on
    
    imagesc((XrgbL));
    %imshow()
    imagesc(XImp,'AlphaData',0.5)
    colormap('jet')
    colorbar;
    function Broadened = broadenRows(Matrix,Factor)
        
            Broadened = zeros(size(Matrix,1)*Factor,size(Matrix,2));
            for i = 1:size(Matrix,1)
                Broadened((i-1)*Factor + 1:i*Factor,:) = repmat(Matrix(i,:),Factor,1);
            end
    end

end

