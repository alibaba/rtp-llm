#!/usr/bin/env python

import datetime
import sys
import os
import time

package_name = '{pkg_prefix}' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.tar.gz'
pangu_root = 'dfs://ea119dfssearch1--cn-shanghai/rtp_pkg/'
dest_root = pangu_root + package_name + '/' # for madrox status
oss_root = 'oss://search-ad/{oss_prefix}/'
oss_http_root = 'http://search-ad.oss-cn-hangzhou-zmf.aliyuncs.com/{oss_prefix}/'

package_json = [
    {
        "type": "ARCHIVE",
        "packageURI": "hdfs://et2prod1/rtp/pkg/hadoop_2.8.0_adp2_3_13-nositexml.tar",
    },
    {
        "type": "ARCHIVE",
        "packageURI": "hdfs://et2prod1/rtp/pkg/search-hdfs-site-2020-08-07-16-50.tar",
    },
    {
        "packageURI": "reg.docker.alibaba-inc.com/isearch/hippo_alios7u2_gcc83_rtp_prod:0.4.9",
        "type": "IMAGE"
    },
    {
        #"packageURI": dest_root + package_name,
        "packageURI": oss_http_root + package_name,
        "type": "ARCHIVE"
    },
]

entity_json = {'package_name': package_name,
               'package_json': package_json}
cmd = 'rm pkgname tmp.tar.gz; cp -L pkg.tar tmp.tar && echo %(pkg_name)s > pkgname && chmod +w tmp.tar && tar -rf tmp.tar pkgname && pigz -p8 tmp.tar && osscmd mp --thread_num=20 tmp.tar.gz oss://search-ad/{oss_prefix}/%(pkg_name)s && rm pkgname tmp.tar.gz' % {'pkg_name':package_name}
print(os.getcwd())
print(cmd)
ret = os.system(cmd)
if ret != 0:
    raise Exception('upload failed')


oss_credentials = open('/home/%s/.osscredentials' % os.environ['USER']).readlines()
accessid = None
accesskey = None
endpoint = None
for line in oss_credentials:
    tokens = [token.strip() for token in line.split('=')]
    if tokens[0] == 'accessid':
        accessid = tokens[1]
    if tokens[0] == 'accesskey':
        accesskey = tokens[1]
    if tokens[0] == 'host':
        endpoint = tokens[1]

print('upload package %s done' % package_name)
