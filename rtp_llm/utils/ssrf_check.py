import socket
import ipaddress
import time
from urllib.parse import urlparse
import logging


logger = logging.getLogger(__name__)


# 获取ip的版本
def is_ipv6(ip):
    try:
        return ipaddress.ip_address(ip).version == 6
    except:
        return False


# 是否是内网IP，是内网返回True，不是内网返回False
def is_intranet(ip):
    intranet_mask = [
                        '10.0.0.0/8',
                        '11.0.0.0/8',
                        '33.0.0.0/8',
                        '127.0.0.0/8',
                        '100.64.0.0/10',
                        '172.16.0.0/12',
                        '192.168.0.0/16'
                    ]

    address = ipaddress.ip_address(ip)

    for mask in intranet_mask:
        network = ipaddress.ip_network(mask)
        if address in network:
            return True

    return False


# 是黑名单域名，返回True；不是黑名单域名，返回False
def is_black_domain(host):
    black_domains = [
            "xiami.net",
            "taobao.org",
            "alibaba.net",
            "alibaba-inc.com",
            "alisoft-inc.com",
            "alidc.net",
            "alipay.net",
            "tbsite.net",
            "aliyun-inc.com",
            "taobao.net",
            "aliloan.net",
            "yunos-inc.com",
            "alipay-inc.com",
            "libaba-inc.com",
            "cainiao-inc.com",
            "alibaba.org",
            "alibank.net",
            "alitrip.net",
            "atatech.org",
            "hitao.net",
            "hupan.net",
            "tmall.net",
            "alimama.net",
            "alifi-inc.com"
    ]
    for black_domain in black_domains:
        if host == black_domain or host.endswith('.' + black_domain):
            logger.error(f'host: {host}. Hit {black_domain} black domain list.')
            return True

    return False


def is_white_domain(host):
    white_domains = [
        "oss-cdn.aliyun-inc.com",
        "alipay-pub.aliyun-inc.com",
        "oss.aliyun-inc.com",
        "csp.yunos-inc.com",
        "aliyunos.com",
        "alicdn.com"
    ]
    for white_domain in white_domains:
        if host == white_domain or host.endswith('.' + white_domain):
            logger.debug(f'host: {host}. Hit {white_domain} white domain list.')
            return True

    return False


# url必须是http://或者https://开头
def check_scheme(url):
    if url.startswith("http://") or url.startswith("https://"):
        return True
    else:
        return False


# 从host中获取ip地址
def get_ip(host):
    try:
        return socket.gethostbyname(host)
    except Exception as e:
        logger.error(f"host: {host}. get_ip function exception: {str(e)}.")
        return None


# 从url中获取host
def get_host(url):
    try:
        return urlparse(url).hostname.lower()
    except Exception as e:
        logger.error("get_host function " + str(e))
        return None


# 危险返回false，安全返回true
def check_ssrf(url):

    # 判断是否为空
    if url is None or len(url) == 0:
        logger.error('url is empty')
        return False

    # 判断协议
    if not check_scheme(url):
        # logger.info(f'url: {url}. Scheme is not http/https.')
        return True

    try:
        host = get_host(url)
        if host is None:
            return False

        # 可能有类似直接这样的url输入：https://[2606:4700:4700::1111]
        if is_ipv6(host):
            logger.debug(f'host: {host} is ipv6. ipv6 is safe.')
            return True

        # 如果是白名单域名，认为安全
        if is_white_domain(host):
            return True

        ip = get_ip(host)
        if ip is None:
            return False

        # 如果是ipv6，认为安全，暂无判断ipv6
        if is_ipv6(ip):
            logger.debug(f'{ip} is ipv6. ipv6 is safe.')
            return True

        # 如果是黑名单域名，返回危险
        if is_black_domain(host):
            return False

        if is_intranet(ip):
            logger.error(f'host: {host}. gethostbyname ip {ip} hit intranet ip.')
            return False

    # 有异常认为危险
    except Exception as e:
        logger.info(str(e))
        return False

    return True