function ImageDispGrey(XRaw)

    XRaw = flipud(XRaw);
    scaledX = rescale(XRaw,100,400,'InputMin',min(XRaw,[],2),'InputMax',max(XRaw,[],2));
    Xrgb = [];
    Factor=1;
    Xrgb(:,:,1) = XRaw;
    Xrgb(:,:,2) = XRaw;
    Xrgb(:,:,3) = XRaw;    
    imagesc(XRaw,[50,450]);  
    colormap('gray');
    xlabel("Samples")
    yticks([1,2,3,4])
    yticklabels(["Sub-Sensor1", "Sub-Sensor2", "Sub-Sensor3", "Sub-Sensor4"])
    a = colorbar;
    yline(1.5)
    yline(2.5)
    yline(3.5)

    a.Label.String = "Standardized logarithmic sensor resistance in ln(Ω)";
    clim([min(min(XRaw)),max(max(XRaw))])
    function Broadened = broadenRows(Matrix,Factor)
        
            Broadened = zeros(size(Matrix,1)*Factor,size(Matrix,2));
            for i = 1:size(Matrix,1)
                Broadened((i-1)*Factor + 1:i*Factor,:) = repmat(Matrix(i,:),Factor,1);
            end
    end

end

