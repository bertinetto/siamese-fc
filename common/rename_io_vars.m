% -------------------------------------------------------------------------------------------------------------------------
function rename_io_vars(net, in, out)
% -------------------------------------------------------------------------------------------------------------------------
    old_in = net.getInputs();
    old_out = net.getOutputs();
    assert(numel(old_in) == 1);
    assert(numel(old_out) == 1);
    net.renameVar(old_in{1}, in);
    net.renameVar(old_out{1}, out);
end