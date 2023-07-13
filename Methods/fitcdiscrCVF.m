function XTable = fitcdiscrCVF(XTable)
    % Used to make sure stride is smaller then kernel
    if XTable.stride > XTable.kernel
       XTable.stride = XTable.kernel-1;
    end
end