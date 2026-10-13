"""Manual PhaNet bundle import and verified official third-party downloads."""
from dataclasses import asdict
from pathlib import Path
import json
import shlex
import shutil
import subprocess
import tarfile

from phanet.annotation import PrepareConfig
from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, read_json
from phanet.resources import _download, _lock, _regular, _verify

# Exact upstream files used by the supported reference/model release.
DOWNLOADS = {
    "esm/esm2_t33_650M_UR50D.pt": {
        "bytes": 2604537549,
        "sha256": "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
    },
    "esm/esm2_t33_650M_UR50D-contact-regression.pt": {
        "bytes": 3687,
        "sha256": "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt"
    },
    "prott5/config.json": {
        "bytes": 546,
        "sha256": "899fe63bf0178fd23583eca238ade81d56a1779c344741f391c3902855c64b77",
        "url": "https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/973be27c52ee6474de9c945952a8008aeb2a1a73/config.json"
    },
    "prott5/pytorch_model.bin": {
        "bytes": 11275562724,
        "sha256": "c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9",
        "url": "https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/973be27c52ee6474de9c945952a8008aeb2a1a73/pytorch_model.bin"
    },
    "prott5/spiece.model": {
        "bytes": 237990,
        "sha256": "74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324",
        "url": "https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/973be27c52ee6474de9c945952a8008aeb2a1a73/spiece.model"
    },
    "prott5/tokenizer_config.json": {
        "bytes": 24,
        "sha256": "a5064475340d80a560463f49921407d901eef3e89667a931eb81a186633eff4b",
        "url": "https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/973be27c52ee6474de9c945952a8008aeb2a1a73/tokenizer_config.json"
    },
    "prott5/special_tokens_map.json": {
        "bytes": 1786,
        "sha256": "4720c0fddbe4c5991334f85ad7073d9bd0a294a8ba4641a2f8dab614ca825949",
        "url": "https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/973be27c52ee6474de9c945952a8008aeb2a1a73/special_tokens_map.json"
    },
    "tmvec/upstream.ckpt": {
        "url": "https://ndownloader.figshare.com/files/49181533",
        "bytes": 409544267,
        "sha256": "5a45f3a7a72431a75d27f60376a17d9dff5a27d830cf4dcafa73861002e770a9"
    }
}
MMSEQS_COMMIT = 'fed69f84d98a2f419693c3b41e0bdb5ad60f7aaa'
MMSEQS_SOURCE = {
    'url': 'https://codeload.github.com/soedinglab/MMseqs2/tar.gz/' + MMSEQS_COMMIT,
    'bytes': 13403978,
    'sha256': '23c9a38c8107bec791222e06f8b9aea640478333c6146412ee0f1b6b8c25758b',
}
TMVEC_ARCH = dict(d_model=1024, nhead=4, num_layers=4, dim_feedforward=2048,
                 out_dim=512, dropout=.1, activation='relu')


def checked_path(path):
    path = Path(path).expanduser().absolute()
    # Refuse symlink parents as well, including a symlink to a not-yet-created root.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'Resource directories must not contain symlinks: {path}')
    return path.resolve()


def directory(path):
    path = checked_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_file(path, data):
    """Create a configuration or reuse an identical existing file."""
    _regular(path)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Existing configuration differs: {path}; choose a new --out directory')
    else:
        with path.open('xb') as stream:
            stream.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def verify_bundle(bundle, method, progress):
    root = Path(bundle).expanduser().absolute()
    if method not in ('glm', 'network', 'combined'):
        raise ValueError('Invalid resource method')
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Provide the extracted PhaNet resource folder, without symlinks')
    _regular(root/'RESOURCE_MANIFEST.json')
    meta = read_json(root/'RESOURCE_MANIFEST.json')
    if not isinstance(meta, dict):
        raise ValueError('Invalid PhaNet resource manifest')
    if meta.get('schema_version') != 1 or meta.get('categories') != list(CATEGORIES):
        raise ValueError('Expected a nine-category PhaNet resource bundle')
    from phanet.network.resources import ARRAYS
    needed = set()
    if method in ('glm','combined'):
        needed |= {'glm/model.json','glm/weights.pt'}
    if method in ('network','combined'):
        needed |= {'network/'+x+'.npy' for x in ARRAYS}
        needed |= {'network/network.json','network/families.json','network/references.faa','profiles/phrog_v4.hmm'}
    records = meta.get('files', {})
    if not isinstance(records, dict) or not needed <= records.keys():
        raise ValueError('PhaNet resource bundle is incomplete for the selected method')
    for name in sorted(needed):
        rec = records[name]; path = root/name
        if (not isinstance(rec, dict) or type(rec.get('bytes')) is not int
                or rec['bytes'] < 0 or not isinstance(rec.get('sha256'), str)
                or len(rec['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in rec['sha256'])):
            raise ValueError(f'Invalid resource record: {name}')
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError(f'Symlink resource: {name}')
        progress(f'Checking PhaNet resource: {name}')
        _verify(path, rec['bytes'], rec['sha256'])
    if method in ('glm','combined'):
        model = read_json(root/'glm/model.json')
        if (not isinstance(model, dict) or model.get('categories') != list(CATEGORIES) or model.get('weights') != 'weights.pt'
                or model.get('weights_sha256') != records['glm/weights.pt']['sha256']):
            raise ValueError('GLM manifest and weights disagree')
    if method in ('network','combined'):
        net = read_json(root/'network/network.json')
        expected_files = {n+'.npy' for n in ARRAYS} | {'families.json', 'references.faa'}
        if (not isinstance(net, dict) or net.get('categories') != list(CATEGORIES)
                or net.get('mmseqs_version') != MMSEQS_COMMIT
                or not isinstance(net.get('files'), dict)
                or set(net['files']) != expected_files or not isinstance(net.get('models'), dict)):
            raise ValueError('Unsupported reference network version')
        if net.get('phrog_sha256') != records['profiles/phrog_v4.hmm']['sha256']:
            raise ValueError('Reference and PHROG resource disagree')
        for name, rec in net['files'].items():
            if records.get('network/'+name) != rec:
                raise ValueError('Reference and outer resource manifests disagree')
    return root.resolve(), meta


def check_mmseqs(binary):
    found = shutil.which(str(binary))
    if not found:
        raise FileNotFoundError('Provide matching --mmseqs, or use --build-mmseqs with CMake and a C++ compiler')
    value = subprocess.run([found,'version'],capture_output=True,text=True,check=True).stdout.strip()
    if value != MMSEQS_COMMIT:
        raise ValueError('MMseqs2 version differs from the reference; use --build-mmseqs or the matching executable')
    return str(Path(found).resolve())


def build_mmseqs(root, *, threads, offline, progress):
    binary = root/'mmseqs/bin/mmseqs'
    if binary.is_file():
        return check_mmseqs(binary)
    for name in ('cmake','c++'):
        if shutil.which(name) is None:
            raise FileNotFoundError(f'MMseqs2 source build requires {name}')
    cache = directory(root/'.downloads'); archive = cache/'mmseqs-source.tar.gz'
    if offline:
        _verify(archive, MMSEQS_SOURCE['bytes'], MMSEQS_SOURCE['sha256'])
    else:
        _download(MMSEQS_SOURCE['url'],archive,MMSEQS_SOURCE['bytes'],MMSEQS_SOURCE['sha256'],progress)
    # Retain failed build logs in the build directory.
    work = root/'mmseqs-build'
    work.mkdir(exist_ok=False)
    source = work/'source'; source.mkdir()
    prefix = 'MMseqs2-'+MMSEQS_COMMIT
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar:
            path = Path(item.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != prefix:
                raise ValueError('Unsafe MMseqs2 source archive path')
            target = source.joinpath(*path.parts[1:])
            if item.isdir():
                target.mkdir(parents=True,exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(item) as inp, target.open('xb') as out:
                    shutil.copyfileobj(inp,out)
                target.chmod(0o755 if item.mode & 0o111 else 0o644)
            else:
                raise ValueError('Unsupported link or special file in MMseqs2 source archive')
    cmds=[['cmake','-S',str(source),'-B',str(work/'build'),'-DCMAKE_BUILD_TYPE=Release',
           '-DCMAKE_INSTALL_PREFIX='+str(root/'mmseqs'),'-DVERSION_OVERRIDE='+MMSEQS_COMMIT],
          ['cmake','--build',str(work/'build'),'--parallel',str(threads)],
          ['cmake','--install',str(work/'build')]]
    progress('Building matching MMseqs2; logs in '+str(work))
    for i, cmd in enumerate(cmds):
        with (work/f'{i}.log').open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    return check_mmseqs(binary)


def setup_resources(bundle, external, output, *, method='combined', traditional_config,
                    mmseqs='mmseqs', build=False, threads=4, offline=False, progress=print):
    """Install model resources and write annotation configuration."""
    if method not in ('glm','network','combined') or type(threads) is not int or threads < 1:
        raise ValueError('Invalid method or build thread count')
    # Validate local prerequisites before any download or build.
    config_path = Path(traditional_config).expanduser().resolve()
    traditional = PrepareConfig.load(config_path)
    own, _ = verify_bundle(bundle,method,progress)
    root = checked_path(external)
    out = checked_path(output)
    for left, right in ((own, root), (own, out), (root, out)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError('Keep bundle, external resources and configuration in separate directories')
    root = directory(root)
    out = directory(out)
    with _lock(root):
        binary = None
        if method != 'glm':
            binary = build_mmseqs(root,threads=threads,offline=offline,progress=progress) if build else check_mmseqs(mmseqs)
        names = [n for n in DOWNLOADS if (n.startswith('esm/') and method!='network') or (not n.startswith('esm/') and method!='glm')]
        required = sum(DOWNLOADS[n]['bytes'] for n in names if not (root/n).exists())
        if shutil.disk_usage(root).free < required:
            raise OSError(f'Insufficient space for third-party resources: need {required/1e9:.1f} GB')
        for name in names:
            record = DOWNLOADS[name]; path = root/name; directory(path.parent)
            if offline:
                _verify(path,record['bytes'],record['sha256'])
            else:
                _download(record['url'],path,record['bytes'],record['sha256'],progress)
        if method != 'glm':
            tmmeta={'schema_version':1,'model':'tm-vec-cath-large','weights':'upstream.ckpt',
                    'weights_sha256':DOWNLOADS['tmvec/upstream.ckpt']['sha256'],'architecture':TMVEC_ARCH}
            stable_file(root/'tmvec/model.json',json_bytes(tmmeta))
            from phanet.representations.tmvec import model_identity
            expected = read_json(own/'network/network.json')['models']
            if model_identity(root/'tmvec',root/'prott5') != expected:
                raise ValueError('Downloaded models differ from the reference representation contract')
        outputs = {'phanet.json':json_bytes({'schema_version':1,**asdict(traditional)})}
        variables={'PHANET_CONFIG':str(out/'phanet.json')}
        if method != 'network':
            variables.update(PHANET_MODEL=str(own/'glm'),PHANET_ESM_MODEL=str(root/'esm/esm2_t33_650M_UR50D.pt'))
        if method != 'glm':
            network={'schema_version':1,'reference':str(own/'network'),'profiles':str(own/'profiles/phrog_v4.hmm'),
                     'tmvec_model':str(root/'tmvec'),'prott5':str(root/'prott5'),'mmseqs':binary}
            outputs['network.json']=json_bytes(network)
            variables['PHANET_NETWORK_CONFIG']=str(out/'network.json')
        clear = 'unset PHANET_MODEL PHANET_ESM_MODEL\n' if method == 'network' else 'unset PHANET_NETWORK_CONFIG\n' if method == 'glm' else ''
        outputs['activate.sh']=(clear + ''.join(f'export {k}={shlex.quote(v)}\n' for k,v in variables.items())).encode()
        receipt={'schema_version':1,'method':method,'bundle_manifest_sha256':digest_file(own/'RESOURCE_MANIFEST.json'),
                 'external_files':{n:DOWNLOADS[n] for n in names},'configuration':list(outputs)}
        outputs['RESOURCES.json'] = json_bytes(receipt)
        # Detect conflicts before writing any config file.
        for name,data in outputs.items():
            path=out/name;_regular(path)
            if path.exists() and path.read_bytes()!=data:
                raise FileExistsError(f'Existing configuration differs: {path}; use a fresh --out directory')
        for name,data in outputs.items():stable_file(out/name,data)
        progress('Resources ready. Source '+str(out/'activate.sh')+' before running phanet annotate.')
        return receipt
