function ImageDisp(XImp, XRaw)
    if size(XImp,1) ~= size(XRaw,1)
        %ToDo
        XImp = [XImp(1:1440);XImp(1441:2880);XImp(2881:4320);XImp(4321:5760)];
    end
    XImp = flipud(XImp);
    XRaw = flipud(XRaw);
    scaledX = rescale(XRaw,0.1,0.9,'InputMin',min(XRaw,[],2),'InputMax',max(XRaw,[],2));
    Xrgb = [];
    Factor=1;
    Xrgb(:,:,1) = broadenRows(scaledX,Factor);
    Xrgb(:,:,2) = broadenRows(scaledX,Factor);
    Xrgb(:,:,3) = broadenRows(scaledX,Factor);
    figure()
    hold on
    
    imagesc((Xrgb));
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

