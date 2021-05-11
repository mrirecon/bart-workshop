function  titlef(str)

if (isnumeric(str))
    str = sprintf('Iteration: %d',str);
end
title(str,'FontSize',14);