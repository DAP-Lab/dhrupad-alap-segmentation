function parsave(fname,data,varargin)
save(fname, '-struct', 'data', varargin{:});
end