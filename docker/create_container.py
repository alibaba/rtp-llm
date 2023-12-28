#!/bin/python
import sys
import os
import getpass
import subprocess
import urllib.request
import json
import argparse
import distutils.spawn

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE = 'registry.cn-hangzhou.aliyuncs.com/havenask/rpt_llm'

TAG='0.0.1'

def cmd_exist(name):
    return distutils.spawn.find_executable(name) is not None
DOCKER = 'docker' if cmd_exist('docker') else 'pouch'

class UserAbort(Exception):
    """User Abort."""
    pass

class BaseExecutor(object):
    def __init__(self):
        self.debugFlag = False

    def debug(self, enable=True):
        self.debugFlag = enable

    def execRet(self, cmd):
        if self.debugFlag:
            print(cmd)
        execcmd = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        out, err = execcmd.communicate()
        return execcmd.returncode, out, err

    def execCMD(self, cmd):
        if self.debugFlag:
            print(cmd)
        if 0 != os.system(cmd):
            raise Exception('exec cmd: [%s] failed!' % cmd)

class ImageInfo(BaseExecutor):
    def __init__(self):
        BaseExecutor.__init__(self)
        self.imageType = 'dev'
        self.enableFilter = True
        self.imageName = IMAGE
        self.tagName = TAG

    def _getLabels(self):
        return {
            'com.search.type': self.imageType
        }

    def setImageType(self, typeName):
        self.imageType = typeName

    def setFilter(self, enable=True):
        self.enableFilter = enable

    def containerLabelStr(self):
        str = ''
        labels = self._getLabels()
        for l in labels:
            str += ' --label %s=%s ' % (l, labels[l])
        return str

    def containerFilterStr(self):
        str = ''
        labels = self._getLabels()
        for l in labels:
            str += ' -f label=%s=%s ' % (l, labels[l])
        return str

    def userFilterStr(self):
        if not self.enableFilter:
            return ''
        return self.containerFilterStr()

    def getImage(self):
        return self.imageName + ':' + self.tagName

    def setImageName(self, name):
        self.imageName = name

    def setTagName(self, name):
        self.tagName = name

class ContainerInfo(ImageInfo):
    def __init__(self):
        ImageInfo.__init__(self)

    def getAllContainers(self, template, filterStr=None):
        if filterStr is None:
            filterStr = self.userFilterStr()
        if DOCKER == "pouch":
            cmd = ("%s ps -a %s | awk '{print $1}'") % (
                DOCKER, filterStr)
        else:
            cmd = ('%s ps -a --format "table %s" %s') % (
                DOCKER, template, filterStr)
        ret, out, err = self.execRet(cmd)
        if ret != 0:
            raise Exception('list container failed.')
        lines = (out.split(b"\n"))
        print("lines = ", lines)
        records = []
        for line in lines:
            if not line:
                continue
            records.append(str(line.decode("utf-8")))
        print("records = ", records)
        return records[0], records[1:]

    def getContainerDetails(self, filterStr=None):
        return self.getAllContainers(
            '{{.Names}}\t{{.ID}}\t{{.Image}}\t' +
            '{{.CreatedAt}}\t{{.Status}}\t{{.Size}}', filterStr)

    def isDevContainer(self, name):
        header, results = self.getAllContainers('{{.Names}}', self.containerFilterStr())
        if name in results:
            return True
        return False

    def choose(self, multi=True):
        header, results = self.getAllContainers('{{.Names}}\t{{.Status}}\t{{.Image}}')
        if not results:
            print('no container to choose.')
            raise UserAbort('')
        selects = {}
        while True:
            print('')
            print(' NUM  %s' % header)
            num = 0
            for result in results:
                if num in selects:
                    tag = '#'
                else:
                    tag = ' '
                print('%s%3d  %s' % (tag, num, result))
                num += 1
            option = raw_input('choose a number[0-%d] (q to exit, c to cont): ' % (num-1))
            subops = option.split()
            for subop in subops:
                if subop == 'q':
                    raise UserAbort('')
                if subop == 'c':
                    if not selects:
                        print('nothing selected.')
                        raise UserAbort('')
                    name = list(selects.values())
                    print('choose container: %s' % ' '.join(name))
                    return name
                try:
                    cop = int(subop)
                    if cop < 0:
                        raise Exception('')
                    if cop in selects:
                        del selects[cop]
                    else:
                        selects[cop] = results[cop].split()[0]
                        if not multi:
                            name = [selects[cop]]
                            print('choose container: %s' % ' '.join(name))
                            return name
                except:
                    print('invalid option %s' % subop)

class CMDBase(ContainerInfo):
    def __init__(self, name, argStr, helpStr):
        ContainerInfo.__init__(self)
        self.myName = name
        self.argStr = argStr
        self.helpStr = helpStr

    def getCMDName(self):
        return self.myName

    def getArgStr(self):
        return self.argStr

    def getHelpStr(self):
        return self.helpStr

    def run(self, args):
        return None

    def doRun(self, args):
        return self.run(args)

    def isCmd(self, cmd, args):
        if cmd == self.myName:
            return True

class CreateCMD(CMDBase):
    def __init__(self, name='create'):
        CMDBase.__init__(self, name, '[container]s',
                         'create dev containers, it will create your account in the container and mount your home dir into container.')
        self.enable_gpu = False
        self.docker_args = ''

    def createOne(self, cname):
        self.execCMD(DOCKER + ' pull ' + self.getImage())
        ramPath = "/mnt/ram"
        if not os.path.exists(ramPath):
            self.execCMD("sudo mkdir -p " + ramPath)
        if subprocess.call("findmnt %s -t ramfs" % (ramPath), shell=True) != 0:
            self.execCMD("sudo mount -t ramfs -o size=20g ramfs " + ramPath)
            self.execCMD("sudo chmod a+rw " + ramPath)

        user = getpass.getuser()
        uid = os.getuid()
        cmd_template = DOCKER + ' run'
        cmd_template += ' ' + self.docker_args + ' '
        cmd_template += ' --cap-add SYS_ADMIN --device /dev/fuse'
        cmd_template += ' -v /ssd/1:/ssd/1'
        cmd_template += ' -v /ssd/2:/ssd/2'
        cmd_template += ' -v /mnt/nas1:/mnt/nas1'
        cmd_template += ' -v /dev/shm:/dev/shm'
        if os.path.exists('/dev/nvidia0') and self.enable_gpu:
            ret, out, err = self.execRet('nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0')
            if ret != 0:
                raise Exception('nvidia-smi fail: %s %s' % (out, err))
            nv_driver_version = str(out.strip())
            # cmd_template += ' --device=/dev/nvidia0:/dev/nvidia0 --device=/dev/nvidia-uvm:/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device=/dev/nvidiactl:/dev/nvidiactl --volume-driver=nvidia-docker --volume=nvidia_driver_' + nv_driver_version + ':/usr/local/nvidia:ro'
            # cmd_template += ' --gpus all --device=/dev/nvidia-uvm:/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device=/dev/nvidiactl:/dev/nvidiactl --volume=nvidia_driver_' + nv_driver_version + ':/usr/local/nvidia:ro'            
            cmd_template += ' --gpus all --device=/dev/nvidiactl:/dev/nvidiactl --volume=nvidia_driver_' + nv_driver_version + ':/usr/local/nvidia:ro'            
        if self.enable_rdma:
            cmd_template += ' --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm'
        cmd_template += ' --name=%s %s -v /home/:/home/ -v /var/run/docker.sock:/var/run/docker.sock -v ' + ramPath + ":" + ramPath
        if DOCKER == 'pouch':
            cmd_template += ' --net=host -dt'
        else:
            cmd_template += ' --net=host -dit'
        cmd_template += ' %s /bin/bash'
        cmd = cmd_template % (cname, self.containerLabelStr(), self.getImage())
        self.execCMD(cmd)

        cmd = '%s exec -i %s groupadd sdev' % (DOCKER, cname)
        code, out, err = self.execRet(cmd)
        if code != 0 and 'already exists' not in err:
            raise Exception('cmd[%s] failed, code[%d] out[%s] err[%s]' %
                            (cmd, code, out, err))

        # cmd = '%s exec -i %s bash -c "echo \\"%%sdev ALL=(ALL) NOPASSWD: ALL\\" | sudo tee /etc/sudoers"' % (
        #     DOCKER, cname)
        # code, out, err = self.execRet(cmd)
        # if code != 0:
        #     raise Exception('cmd[%s] failed, code[%d] out[%s] err[%s]' %
        #                     (cmd, code, out, err))

        cmd = ('%s exec -i %s /usr/sbin/useradd -MU -G sdev -u %s %s') % (
            DOCKER, cname, uid, user)
        code, out, err = self.execRet(cmd)
        if code != 0 and 'already exists' not in err:
            raise Exception('cmd[%s] failed, code[%d] out[%s] err[%s]' %
                            (cmd, code, out, err))

    def run(self, args):
        parser = argparse.ArgumentParser(description='ddev create')
        parser.add_argument('cname')
        parser.add_argument('--gpu', action='store_true')
        parser.add_argument('--host_cuda', action='store_true',
                            help="mount nvidia device into container, but not cuda libraries by default, host_cuda will mount host cuda libraries into container, must used with --gpu.")
        parser.add_argument('--image', default=IMAGE)
        parser.add_argument('--tag', default=TAG)
        parser.add_argument('--docker_args', default='')
        parser.add_argument('--rdma', action='store_true', help="mount rdma device into container")
        args = parser.parse_args(args)
        self.setImageName(args.image)
        self.setTagName(args.tag)
        self.enable_gpu = args.gpu
        self.enable_rdma = args.rdma
        self.host_cuda = args.host_cuda
        self.docker_args = args.docker_args
        self.createOne(args.cname)

class CreateWithGpuCMD(CreateCMD):
    def __init__(self):
        super(CreateWithGpuCMD, self).__init__(name='create_with_gpu')
        self.enable_gpu = True

class CreateWithRdmaCMD(CreateCMD):
    def __init__(self):
        super(CreateWithRdmaCMD, self).__init__(name='create_with_rdma')
        self.enable_rdma = True

class LsCMD(CMDBase):
    def __init__(self):
        CMDBase.__init__(self, 'ls', '', 'ls dev containers, with option all to show all containers.')

    def run(self, args):
        header, results = self.getContainerDetails()
        print(header)
        print('\n'.join(results))

class SimpleCMD(CMDBase):
    def __init__(self, name):
        CMDBase.__init__(self, name, '[container]s', 'do command on containers, same as the corresponding command in docker. with option all to select all containers.')

    def run(self, args):
        if len(args) == 0:
            args = self.choose()
        cmd = '%s %s %s' % (DOCKER, self.getCMDName(), ' '.join(args))
        self.execCMD(cmd)

class EnterCMD(CMDBase):
    def __init__(self):
        CMDBase.__init__(self, 'enter', '[container]', 'enter container using nsenter, needs root privilege. with option all to select all containers.')

    def run(self, args):
        if len(args) == 0:
            args = self.choose(False)
        user = getpass.getuser()
        cname = args[0]
        if self.isDevContainer(cname):
            extra = ['/usr/bin/su', user]
        else:
            extra = []
        ret, out, err = self.execRet(
            '%s inspect --format "{{ .State.Pid }}" %s' % (DOCKER, cname))
        if ret != 0:
            raise Exception('container not found.')
        cmd = ['/usr/bin/sudo', 'nsenter', '--target', out.strip().decode("utf-8"),
               '--mount', '--uts', '--ipc', '--net', '--pid']
        cmd.extend(extra)
        print("cmd = ", cmd)
        all_strings = list(map(str, cmd))
        print("all_strings = ", all_strings)
        self.execCMD(' '.join(all_strings))

class DEnterCMD(CMDBase):
    def __init__(self):
        CMDBase.__init__(self, 'denter', '[container]', 'enter container using docker exec, do not need root privilege. with option all to select all containers.')

    def run(self, args):
        if len(args) == 0:
            args = self.choose(False)
        user = getpass.getuser()
        cname = args[0]
        if self.isDevContainer(cname):
            extra = ['/usr/bin/su', user]
        else:
            extra = ['bash']
        cmd = [DOCKER, 'exec', '-it', cname]
        cmd.extend(extra)
        self.execCMD(' '.join(cmd))

class ClearNoneCMD(CMDBase):
    def __init__(self):
        CMDBase.__init__(self, 'clearnone', '', 'clear <none> images')

    def run(self, args):
        self.execCMD(DOCKER + ' rmi $(docker images --filter "dangling=true" -q --no-trunc)')

class HippoBase(CMDBase):
    def __init__(self, name, argStr, helpStr):
        argStr = '%s %s' % (name, argStr)
        CMDBase.__init__(self, 'hippo', argStr, helpStr)
        self.subCMD = name

    def getOneHippoContainers(self, httpPort, arpcPort):
        containers = []
        url = 'http://127.0.0.1:%s/SlaveService/getSlotDetails' % httpPort
        req = urllib2.Request(url)
        response = urllib2.urlopen(req)
        result = response.read()
        info = json.loads(result)
        for slot in info['slaveDetail']['slots']:
            container = {}
            container['workdir'] = slot['workDir']
            container['app'] = slot['info']['applicationId']
            container['slotid'] = slot['info']['slotId']
            container['contid'] = '-'
            for resource in slot['info']['slotResource']['resources']:
                if resource['name'] == 'T4':
                    container['contid'] = 'HIPPO_%s_%d' % (
                        arpcPort, container['slotid'])
            container['pids'] = []
            for process in slot['info']['processStatus']:
                container['pids'].append(process['pid'])
            slot['contid'] = container['contid']
            container['raw'] = slot
            containers.append(container)
        return containers

    def getAllHippoContainers(self):
        ret, out, err = self.execRet(
            'ps ax | grep hippo_slave | ' +
            r'sed -n "s/.*-p \([0-9]*\).*-P \([0-9]*\).*/\1 \2/p"')
        if ret != 0:
            raise Exception('find hippo slave error')
        lines = out.split('\n')
        containers = []
        for line in lines:
            if not line:
                continue
            port = line.split()
            if len(port) < 2:
                continue
            containers.extend(self.getOneHippoContainers(port[1], port[0]))
        return containers

    def doRun(self, args):
        args = args[1:]
        CMDBase.doRun(self, args)

    def isCmd(self, cmd, args):
        if not CMDBase.isCmd(self, cmd, args):
            return False
        if len(args) > 0 and args[0] == self.subCMD:
            return True
        return False

class HippoLsCMD(HippoBase):
    def __init__(self):
        HippoBase.__init__(self, 'ls', '', 'ls hippo slots')

    def run(self, args):
        containers = self.getAllHippoContainers()
        print(('%8s  %20s  %-40s') % ('SLOTID', 'CONTAINER', 'APPID'))
        for container in containers:
            print (('%8d  %20s  %-40s') % (
                container['slotid'], container['contid'], container['app']))

class HippoInfoCMD(HippoBase):
    def __init__(self):
        HippoBase.__init__(self, 'info', 'slotnum', 'detail info for a slot')

    def run(self, args):
        if len(args) == 0:
            raise Exception('slotid not found.')
        slotid = int(args[0])
        containers = self.getAllHippoContainers()
        for container in containers:
            if container['slotid'] == slotid:
                print(json.dumps(container['raw'], sort_keys=True,
                                 indent=4, separators=(',', ': ')))
                break

class HippoFromPidCMD(HippoBase):
    def __init__(self):
        HippoBase.__init__(self, 'frompid', 'pid', 'find hippo slot from pid')

    def run(self, args):
        if len(args) == 0:
            raise Exception('pid not found.')
        pid = int(args[0])
        containers = self.getAllHippoContainers()
        for container in containers:
            if pid in container['pids']:
                print(json.dumps(container['raw'], sort_keys=True,
                                 indent=4, separators=(',', ': ')))
                break

class RmCMD(CMDBase):
    def __init__(self):
        CMDBase.__init__(self, 'rm', '', 'stop and rm dev containers')

    def run(self, args):
        SimpleCMD('stop').run(args)
        SimpleCMD('rm').run(args)

def thelp(cmds):
    helpInfo = '''docker tools to manage dev and product containers
Commands:
'''
    myName = 'ddev'
    for c in cmds:
        helpInfo += '''    %s [debug] [all] %s %s
        %s
''' % (myName, c.getCMDName(), c.getArgStr(), c.getHelpStr())

    print(helpInfo)

def main():
    debug = True
    filter = True
    args = sys.argv[1:]
    cmds = [
        CreateCMD(), CreateWithGpuCMD(), LsCMD(), SimpleCMD('start'), SimpleCMD('stop'),
        RmCMD(), EnterCMD(), DEnterCMD(), ClearNoneCMD(),
        HippoLsCMD(), HippoInfoCMD(), HippoFromPidCMD(),
    ]

    while True:
        if len(args) == 0:
            return thelp(cmds)
        cmd = args[0]
        args = args[1:]
        if cmd == 'debug':
            debug = True
        elif cmd == 'all':
            filter = False
        else:
            for c in cmds:
                if c.isCmd(cmd, args):
                    c.debug(debug)
                    c.setFilter(filter)
                    c.doRun(args)
                    return

if __name__ == "__main__":
    try:
        main()
    except UserAbort as e:
        pass
    sys.exit(0)
