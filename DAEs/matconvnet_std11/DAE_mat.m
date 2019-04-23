function rec = DAE_mat(x, net)

rec = vl_simplenn(net, x,[],[],'conserveMemory',true,'mode','test');
rec = x + double(rec(end).x);
