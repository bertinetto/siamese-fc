%VSIZE	disect a variable and show the size of its members
%
%	VSIZE shows the size and layout of
%	-  named   variables	a variable as seen in the workspace
%	-  unnamed variables	part of a variable
%	and their
%	-  containers		CELLs/STRUCTs
%	-  members		other data types
%
%	see also: who, whos, ndims, size, numel, length
%
%SYNTAX
%-------------------------------------------------------------------------------
%		    VSIZE(VAR1,OPT1,VAR2,...,OPTn,...,VARn)
%		P = VSIZE(...)
%
%INPUT
%-------------------------------------------------------------------------------
% VAR	:	variable of any data type
%		- named		top level name, eg, C, S
%		- unnamed	part of a variable, eg, C{4}, S.a(4).b
%
% OPT		argument	description
% -------------------------------------------------------------------
% -r	:	none		show top-level variable(s)   only
%				 DL = 1
% -f	:	none		show top-level fieldname(s)  only
%				 DL = 2
%				 suitable for STRUCTs
% -c	:	none		show container classes       only
%				 DL = [def]
%
% -d	:	DL		recurse to depth level DL   [def = inf]
% -bs	:	BS		show sizes as BS: B K M G   [def = B]
%				 BS may be lower case and
%				 followed by a trailing
%				 string, eg,
%				 'bytes', 'Kb', 'Giga'
%
%OUTPUT
%-------------------------------------------------------------------------------
% P	:	information about each container/member of a variable
% col 1	:	field descriptor
% col 2	:	size
% col 3	:	class
% col 4	:	member descriptor			NDIM:SIZE:CLASS
% col 5	:	depth  level of container/member
% col 6	:	number of input argument
% col 7	:	number of container			cell|struct
% col 8	:	number of member			all other objects
%		col 5-8 according to depth level DL
%
%EXAMPLE
%-------------------------------------------------------------------------------
% 		clear v;
% 		v(1).a{1}=sparse(magic(3)+2i*magic(3));
% 		v(2).a{2}={struct('FA',{'a','bb'},'FB',{magic(5),{}})};
% 		v(2).b{2}=@(x) sind(x);
%
% 		vsize(v,'-f');	% or: vsize(v,'-d',2);
% % -------------------------
% %       1346       1346 B *   v = 2:1x2:struct.(2)
% % CELL -----        256 B     v[].a = 2:1x1:cell
% % CELL -----        698 B     v[].a = 2:1x2:cell
% %       1346          0 B -   v[].b = 2:0x0:double
% % CELL -----         80 B     v[].b = 2:1x2:cell
%
% 		vsize(v,1i,'-c','-bs','Bytes');
% % -----------------------------
% %       1346       1346 Bytes *   v = 2:1x2:struct.(2)
% % CELL -----        256 Bytes     v[].a = 2:1x1:cell
% % CELL -----        698 Bytes     v[].a = 2:1x2:cell
% % CELL -----        634 Bytes     v[].a{} = 2:1x1:cell
% % STRUCT ---        574 Bytes     v[].a{}{} = 2:1x2:struct.(2)
% % CELL -----          0 Bytes     v[].a{}{}[].FB = 2:0x0:cell
% % CELL -----         80 Bytes     v[].b = 2:1x2:cell
% % -----------------------------
% %         16         16 Bytes *   * = 2:1x1:double.complex
%
% 		e=vsize(v,v(1).a{1});
% -------------------------
%       1346       1346 B *   v = 2:1x2:struct.(2)
% CELL -----        256 B     v[].a = 2:1x1:cell
%       1150        196 B -   v[].a{} = 2:3x3:double.sparse.complex
% CELL -----        698 B     v[].a = 2:1x2:cell
%       1150          0 B -   v[].a{} = 2:0x0:double
% CELL -----        634 B     v[].a{} = 2:1x1:cell
% STRUCT ---        574 B     v[].a{}{} = 2:1x2:struct.(2)
%       1148          2 B -   v[].a{}{}[].FA = 2:1x1:char
%       1144          4 B -   v[].a{}{}[].FA = 2:1x2:char
%        944        200 B -   v[].a{}{}[].FB = 2:5x5:double
% CELL -----          0 B     v[].a{}{}[].FB = 2:0x0:cell
%        944          0 B -   v[].b = 2:0x0:double
% CELL -----         80 B     v[].b = 2:1x2:cell
%        944          0 B -   v[].b{} = 2:0x0:double
%        928         16 B -   v[].b{} = 2:1x1:function_handle
% -------------------------
%        196        196 B *   * = 2:3x3:double.sparse.complex
%
% % col 1:	top-level size - member size [bytes]
% %		- container sizes are NOT subtracted
% %		- the last number shows the overhead of the containers
% %			argument 1:	928
% %			argument 2:	has no containers|members
% % col 2:	size of the member	bytes or according to -bs BS
% %			### BS * :	top-level
% %			### BS - :	member
% % col 3:	field descriptor = member dimension:size:class
% %			       * :	unnamed input

% created:
%	us	08-Aug-2006
% modified:
%	us	27-May-2010 22:55:07
% localid:	us@USZ|ws-nos-36362|x86|Windows XP|7.10.0.499.R2010a

function	[p,par]=vsize(varargin)

		F=false;
		T=true;

		otbl={
% user		--------------------------------------------------
		'-bs'		1	'B'	'byte size'
		'-c'		0	F	'containers'
		'-d'		1	inf	'search depth'
		'-f'		0	F	'top level fields'
		'-r'		0	T	'top level'
		};

		cc={
			'cell'
			'struct'
		};

	if	nargout
		p=[];
	end
	if	nargin < 1
		help(mfilename);
		return;
	end

% initialize engine
		[p,par,arglst]=VSIZE_set_par(-1,cc,nargin,[],otbl,varargin{:});

	for	i=par.nb:arglst.argn
% tokenize input arguments
		inam=inputname(arglst.argx(i));
		carg=varargin{arglst.argx(i)};
		[p,par]=VSIZE_run(0,inam,p,par,carg);
	end
		p=par.r;
	if	~nargout
		clear p;
	end
end
%-------------------------------------------------------------------------------
function	[p,par]=VSIZE_run(isrec,inam,p,par,varargin)

		[p,par]=VSIZE_set_par(isrec,inam,p,par,[],varargin{:});
	if	isempty(p)
		return;
	end
		[p,par]=VSIZE_get_tok(p,par);
end
%-------------------------------------------------------------------------------
function	[p,par,opo]=VSIZE_set_par(isrec,inam,p,par,otbl,varargin)
% create common parameter list

			par.darg=4;
			par.narg=0;
			par.hasvar=false;
			par.hasanomaly=false;
			par.fnam=[];

	switch	isrec
% recursion
	case	1
			par.opt.cflg=true;
			par.opt.rflg=true;
			par.level=varargin{1};
			par.vnam=varargin{2};
			par.arg=varargin(3:end);
			par.narg=numel(par.arg);
			par.inam=par.vnam;
		if	par.level >= par.rlim
			disp(sprintf('VSIZE> recursion limit reached %5d',par.rlim));
			p=[];
		end
			return;

% input argument
	case	0
			par.opt.cflg=false;
			par.opt.sflg=0;
			par.noname='*';
			par.level=0;
			par.b=0;
			par.inam=par.noname;
		if	~isempty(inam)
			par.inam=inam;
		end
			par.vnam=par.inam;
			par.arg=varargin;
			par.narg=numel(par.arg);
			par.na=par.na+1;
			par.ne=1;
			par.nm=0;

% initialize engine
	case	-1
			opo=VSIZE_getoptn(otbl,varargin{:});

			par.opt=[];
			par.cc=inam;
			par.rlim=get(0,'recursionlimit');
			par.level=0;
			par.mlevel=inf;

			narg=p;
			p=[];
			p.narg=narg;

			opt=opo.opt;
			par.mlevel=opt.dflg;
		if	opt.fflg
			par.mlevel=2;
		end
		if	opt.cflg
			opt.Cflg=true;
			opt.cflg=false;
		else
			opt.Cflg=false;
		end
			opt.sflg=0;
			opo.opt=opt;
			par.opt=opt;
			par.nb=1;
			par.argn=opo.argn;
			par.argx=opo.argx;
			par.na=0;
			par.ne=0;
			par.nm=0;
			par.r={};
			par.fn={};

% - size constant
		switch	lower(par.opt.bsflg(1))
		case	'b'
			par.vs=1;
		case	'k'
			par.vs=1024;
		case	'm'
			par.vs=1024^2;
		case	'g'
			par.vs=1024^3;
		otherwise
			error('\n%s> [-bs] invalid size value [''b'',''k'',''m'',''g'']\n','VSIZE');
		end

% - output format
			nf=3-par.opt.rflg;
			fmt={
				'%3d>%5d/%5d/%5d: %10s %10d%s   %s = %s'	1	45+numel(par.opt.bsflg)-1
				'%% %10s %10d XX %s   %s = %s'			5	25+numel(par.opt.bsflg)-1
				'%% %10d XX %s   %s = %s'			6	14+numel(par.opt.bsflg)-1
			};
			fmt(:,1)=strrep(fmt(:,1),'XX',par.opt.bsflg);
			par.fmt=fmt{nf,1};
			par.fmtb=fmt{nf,2};
			par.d=['% ',repmat('-',1,fmt{nf,3})];

% - get current <whos> entries
%	7.2.0.232	(R2006a)
%	7.3.0.32269	(R2006b)
%	7.7.0.471	(R2008b)
%	7.10.0.499	(R2010a)
			wn={
				'name'
				'size'
				'bytes'
				'class'
%				'global'
%				'sparse'
%				'complex'
				'nesting'
%				'persistent'
%   engine
				'obytes'
			};

			w=VSIZE_whos(par,nan);
			wf=fieldnames(w);
			par.wf=wf(~ismember(wf,wn));
			par.anomaly='2:0x0:anomaly (struct)';

		if	~par.opt.rflg
			disp(par.d);
		end
	end
end
%-------------------------------------------------------------------------------
function	p=VSIZE_getoptn(otbl,varargin)

% fast option parser

		argo=varargin;
		narg=numel(argo);

		ot=otbl(:,1).';
		rex='^([\W\d\_]+)(.+)';
		ff=regexprep(ot,rex,'$2flg');
		nf=numel(ff);

		fn=[
			ff
			otbl(:,3).'
		];

		argx=true(size(argo));
	for	i=1:nf
		ix=strcmp(argo,ot{i});
	if	any(ix)
	if	otbl{i,2}
		no=otbl{i,2};
		id=find(ix);
	if	id(end)+no <= narg
	if	no == 1
		fn(2,i)=argo(id(end)+1);
	else
		fn{2,i}={argo(id(end)+1:id(end)+no)};
	end
		ix(id+1)=true;
	end
	else
		fn{2,i}=~otbl{i,3};
	end
		argx(ix)=false;
	end
	end
		o=struct(fn{:});

		argo(~argx)=[];
		argn=numel(argo);

		p.opt=o;
		p.argn=argn;
		p.argx=find(argx);
end
%-------------------------------------------------------------------------------
function	[p,par]=VSIZE_get_tok(p,par)

% tokenize input arguments

	for	n=1:par.narg
		arg=par.arg{n};
		[p,par]=VSIZE_get_arg(p,par,arg);
		[p,par]=VSIZE_get_fld(p,par,arg);
		[p,par]=VSIZE_get_ent(p,par,arg);
	end
end
%-------------------------------------------------------------------------------
function	[p,par]=VSIZE_get_arg(p,par,arg)

			par.hasvar=false;
	if	isstruct(arg)
			par.fnam=fieldnames(arg);
			par.opt.sflg=par.opt.sflg+1;
		if	par.opt.cflg		&&...
			par.opt.sflg ~= 1
			par.hasvar=true;
			par.opt.sflg=0;
		end
		if	par.opt.sflg==1
			par.hasvar=false;
			par.opt.sflg=par.opt.sflg+1;
		end
	else
			par.opt.sflg=0;
	end

			a=VSIZE_whos(par,arg);
			a.bytes=a.obytes;
		if	~par.level
			par.b=a.bytes;
		end

	if	par.hasvar
			p.nv=numel(arg);
			p.nf=numel(par.fnam);
			p.n=p.nv*p.nf;
			p.as=size(arg);
			p.fr=cell(p.n,1);
			p.fn=cell(p.n,4);
			ix=0;
		for	j=1:p.nf
		for	i=1:p.nv
			ix=ix+1;
			p.fr(ix,1)=par.fnam(j);
			p.fn(ix,1)=par.fnam(j);
		end
		end
	else
			p.nv=1;
			p.nf=numel(par.fnam);
			p.n=1;
			p.as=size(arg);
			p.fn=cell(1,4);
			p.fr={[]};
	end
end
%-------------------------------------------------------------------------------
function	[p,par]=VSIZE_get_fld(p,par,arg)

	for	i=1:p.n
			vx=rem(i-1,p.nv)+1;
	if	par.hasvar
			cf=p.fn{i,1};
		try
			t=arg(vx).(cf);
		catch	%#ok
			disp('VSIZE> unexpected assignment error!');
			keyboard
		end
			p.fn{i}=sprintf('%s.%s',par.inam,p.fn{i});
	else
			t=arg;
			p.fn{i}=sprintf('%s',par.inam);
	end

			w=VSIZE_whos(par,t);
			c=w.class;
		if	isa(t,'struct')
			nfn=numel(fieldnames(t));
			c=sprintf('%s.(%d)',c,nfn);
		end
		for	j=1:numel(par.wf)
		if	w.(par.wf{j})
			c=[c,'.',par.wf{j}];		%#ok
		end
		end

% common size loader
			spc=sprintf('%-1dx',w.size);
			spc=sprintf('%-1d:%s',ndims(t),spc);
			spc=sprintf('%s:%s',spc(1:end-1),c);
			p.fn{i,2}=w.bytes;
			p.fn{i,3}=w.class;
			p.fn{i,4}=spc;

% common size loader
		if	isnumeric(p.as)			&&...
			~sum(p.as)
			par.hasanomaly=true;
			p.fn{i,3}=class(arg);
		end
	end

		if	isempty(p)			||...
			~p.n
			par.hasanomaly=true;
			p.n=numel(par.fnam);
			p.fr={[]};
		for	i=1:p.n
			p.fn(i,:)={[par.inam,'.',par.fnam{i}],0,'anomaly',par.anomaly};
		end
		end
end
%-------------------------------------------------------------------------------
function	[p,par]=VSIZE_get_ent(p,par,arg)

	for	i=1:p.n

			sflg=true;
		if	par.level
% member
		if	~any(ismember(par.cc,p.fn{i,3}))
			par.b=par.b-p.fn{i,2};
			par.db=sprintf('%-1d',par.b);
			par.dlevel='-';
			par.nm=par.nm+1;
		if	par.opt.Cflg
			sflg=false;
		end
		else
% container
			cc=p.fn{i,3};
			par.db='container';
			par.db=sprintf('%-9.9s',upper(cc));
			par.db(numel(cc)+2:10)='-';
			par.dlevel=' ';
			par.ne=par.ne+1;
			par.nm=0;
		end
		else
% top-level variable
			par.db=sprintf('%-1d',par.b);
			par.dlevel='*';
		if	par.opt.rflg
			disp(par.d);
		end
		end

% update/display current structure
			par=VSIZE_show_ent(sflg,p,par,i,...
				par.level,par.na,par.ne,par.nm,par.db,p.fn{i,2},par.dlevel,p.fn{i,1},p.fn{i,4});
			par.hasanomaly=false;
			[p,par]=VSIZE_get_rec(p,par,arg,i);
	end
end
%-------------------------------------------------------------------------------
function	[p,par]=VSIZE_get_rec(p,par,arg,ix)
% resolve recursion

	if	par.opt.rflg				&&...
		p.fn{ix,2}

			par.level=par.level+1;
			vx=rem(ix-1,p.nv)+1;
	switch	p.fn{ix,3}
% - struct
	case	'struct'
			par.opt.sflg=-1;
			sb='';
		if	~isempty(p.fr{ix})
			val=arg(vx).(p.fr{ix});
		else
			val=arg;
		end
		if	numel(val) > 1
			sb='[]';
		end
			val={val};
% - cell
	case	'cell'
			sb='{}';
		if	par.hasvar
			val=arg(vx).(p.fr{ix});
		else
			val=arg;
		end
% - member
	otherwise
			par.level=par.level-1;
			return;

	end	% SWITCH

	if	par.level < par.mlevel
			[tpar,tpar]=VSIZE_run(1,p.fn{ix,3},p,par,...
				par.level,[p.fn{ix},sb],val{:});	%#ok
			par.ne=tpar.ne;
			par.b=tpar.b;
			par.r=tpar.r;
			par.level=par.level-1;
			par.opt.sflg=0;
	end

	end
end
%-------------------------------------------------------------------------------
function	w=VSIZE_whos(par,arg)			%#ok
% common WHOS engine

		w=whos('arg');
		w.obytes=w.bytes;
		w.bytes=round(w.bytes/par.vs);
end
%-------------------------------------------------------------------------------
function	par=VSIZE_show_ent(sflg,p,par,ix,varargin)
% general purpose print routine

		v=[p.fn(ix,:),varargin(1:4)];
		par.r=[par.r;v];
%D		disp(sprintf('SHOW> %5d %10s %s',sflg,varargin{[7,5]}));
	if	sflg
		s=sprintf(par.fmt,varargin{par.fmtb:end});
		disp(s);
	end
end
%-------------------------------------------------------------------------------