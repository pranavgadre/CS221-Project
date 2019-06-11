figure(1)
hist(Y_all, -2:2)
% histHandle = histogram(Y_all)
% figHandle = figure(1);
% axisHandle = figHandle.Children;
% histHandle = axisHandle.Children;
% histHandle.BinEdges = histHandle.BinEdges + histHandle.BinWidth/2;
legend('Ground Truth', 'Neural Net', 'Log. Reg.')
