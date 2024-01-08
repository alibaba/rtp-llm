import os
import sys
import signal
from threading import Thread
import requests
import time
import logging

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))

from maga_transformer.start_server import main as server_main

from py_inference.benchmark.bench_args import BenchArgs
from py_inference.benchmark.test_object import BaseTester, HttpTester
from py_inference.benchmark.benchmark_runner import BenchRunner
from py_inference.benchmark.analyzers.llm_analyzer import LlmAnalyzer


def wait_server_start(server_thread: Thread, port: int):
    start_time = time.time()
    while True:
        time.sleep(10)
        try:
            if not server_thread.is_alive():
                raise SystemExit("Server thread dead!")
            requests.get(f"http://localhost:{port}/status")
            break
        except Exception as e:
            logging.info(f"Waiting server on {port}, used {time.time() - start_time}s: {e}")
            continue


if __name__ == '__main__':
    try:
        os.setpgrp()
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    server_thread = Thread(target=server_main)
    server_thread.start()

    port = int(os.environ["START_PORT"])
    wait_server_start(server_thread, port)
    request = {
        "prompt": "\n\nHuman: 你是谁\n\nAssistant: ",
        "generate_config": {}
    }
    try:
        response = requests.post(f"http://localhost:{port}/", json=request)
        print(response.json())
    except Exception as e:
        print(e)
    time.sleep(5)

    query_prompt = "\n\nHuman: 已知信息：\n\n[1] 云上应用架构指南：完整手册:架构设计约束-高性能计算架构风格：什么是高性能计算架构 高性能计算 (High Performance Computing，HPC）可以聚集计算能力，使用大量CPU/GPU执行并行计算，进行仿真、建模、3D渲染。高性能计算通常使用高速RDMA网络互联的CPU以及GPU等异构加速设备，面向高性能计算、人工智能/机器学习、科学/工程计算、数据分析、音视频处理等应用务。阿里云弹性高性能计算E-HPC支持高性能CPU、异构计算GPU实例的IaaS服务，提供高性能计算软件栈的PaaS服务。高性能计算架构具备以下特征：整体计算任务拆分为离散的小型任务，分布在多个CPU/GPU计算单元中处理。您可以根据工作负载的大小，使用不同型号的计算单元。某些应用的小型任务是独立且并行的。但很多事实，小型任务之间紧密耦合，需要持续进行交互。此时，您可以考虑使用远程直接内存访问（ RDMA）等高速联网技术，提升交互效率。任务的计算不会无限进行，一定可以在有限的时间内完成。您可以在特定的计算高峰时间点，为任务分配大量的计算单元，任务完成后，逐渐将分配的计算单元资源减少为0。应用程序不需要全天候运行，但是必须随时准备好处理计算节点的故障。什么时候使用高性能计算架构 高性能计算主要应用领域有：解决大规模科学问题，例如力学、气动、热力学工程问题，以及天气预报、地形分析和生物制药等应用场景。存储和处理海量数据，数据挖掘、图象处理和基因测序等。耗时长久或重复计算的场景，高效率的并行分布式处理系统可以有效降低计算时长。模拟和数字运算等计算密集型场景。例如一个模拟计算，对于单台计算机而言内存过小无法满足需求，须拆分到多台计算机中完成。高性能计算架构的优点：高性能，高效率。可以同时利用数百或数千个计算单元，更快地完成大规模计算。与其他高性能专用硬件兼容。支持弹性伸缩，可以按需购买计算单元。\n 我在阿里十年的资源管理总结:比如一个4核容器，通过将其规格画像为requests.cpu=2,limits.cpu(最大cpu使用核数)=12,同时在底层cgroup将cfs_quota_us设置为1200000不变，保证业务的资源与其他CPUshare的业务共享12核，如图3-1图3-1CPUShare1VPA与CPU峰值利用率的关系。可能大家会有疑问，VPA并不能直接提高整个集群的CPU峰值利用率呀，为什么要把VPA机制放到提升CPU峰值利用率的范围内讲呢。这其实是个误解，从绝对值上说VPA确实无法直接提升整个资源池的利用率，但是VPA其实一种软降配。软降配能够空余出更多的核来，这样可以用不变的资源池大小，来跑更多的在线业务。从这个视角上来看，他是能够提高CPU峰值利用率的。我们用图3-2CPUshare2更加直观。虽然App1~App3的CPUSharedPool从12核下降到8核，但是相比原来CPUSet时每个App最多使用4核还是提高了上限的。同时经过CPUShare之后，可以多调度一个App5进来跑。只要App5不是空转，就能够提高在线CPU峰值利用率了。图3-2CPUshare2HPAHPA(HorizontalPodAutoscaler)是一种通过扩展pod的数量来进行自动扩缩容的机制。可根据观察到的CPU、内存使用率或自定义度量标准来自动扩展或缩容Pod的数量。HPA不适用于无法缩放的对象，比如DaemonSetHPA控制器会定期调整RC或Deployment的副本数，以使观察到的平均CPU利用率与用户指定的目标相匹配。图3-3HPAHPA解决什么问题。HPA就如同他的定义一样，是进行自动扩缩容用的，它实际上是一个高效的负反馈系统[6]。如果应用负载高了，就自动扩容，如果应用负载低了就自动缩容。在提高CPU峰值利用率方面我们只讨论自动缩容的方面。HPA快扩慢缩。在缩容过程中，生产上曾经做过多次尝试，社区也很难达成一致。目前k8s上默认的参数是扩容3分钟、缩容5分钟。\n\n\n根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。\n\n\n问题：\n高P\n\nAssistant:"

    batch_sizes = [1, 2, 4, 8]
    # batch_sizes = [1]

    for batch_size in batch_sizes:
        query = {
            "prompt_batch": [query_prompt] * batch_size,
            "generate_config": {
                "max_new_tokens": 256,
            }
        }

        bench_args = BenchArgs(
            queries=[query],
            repeat_limit=5,
            concurrent_request_count=1,
        )

        tester = HttpTester(host="localhost", port=port)
        bench_runner = BenchRunner(tester, bench_args, [LlmAnalyzer()])
        bench_runner.run()

        logging.info(f"tested batch size = {batch_size}")

    os.killpg(0, signal.SIGKILL)
    os._exit(0)

    server_thread.join()
