## 📝 进行中的任务
- [ ] 功能开发
  - [✅] worker增加machine_info
  - [✅] master暂时改成int64 request_id走通流程
- [ ] 代码优化
  - [✅] worker response里增加任务的等待时间，方便对比
  - [✅] remote tokenize每次会有connect，看下是否能够复用，现在tokenize成本很高
  - [✅] 修改/worker_status, 把last_schedule_time改成last_schedule_delta，在worker端把schedule没更新的问题处理掉
  - [✅] 修改master日志，把预期结束时间换成预期等待delta
  - [✅] tokenize现在会占用比较长的时间，但是原因是跨机房，得去同一个机房看看, 如果时间长的话需要考虑把tokenize结果复用
  - [ ] 监控补全
  - [ ] 优化master的request_id生成方式, 从int64->string，虽然int64问题也不大，但是string更不会冲突


## 🐛 Bug修复
- [✅] Bug1：并发请求会挂
  - 复现步骤： 将同步worker时间改短以后，并发请求会挂，得看看为啥
  - 优先级：高
  - 原因：load balancer会摘除健康检查不对的worker，在单台机器下处理不过来这么多请求，超时设置短就会被摘掉
  - 解决方案： 加大超时时间和间隔
- [✅] Bug2：报错
  [RANK 0][maga_transformer/cpp/disaggregate/rtpllm_master/estimator/LookupMapImpl.cpp:66][bool rtp_llm::rtp_llm_master::LookupMapImpl::checkConfigValid(int, int, const rtp_llm::rtp_llm_master::SingleConfig&, const rtp_llm::rtp_llm_master::SingleConfig&) const] input or prefix lower bound match failed, expect: [1024:0], actual: [1024:2048]]
  - 复现步骤：起服务就有
  - 优先级：高
  - 原因： 代码写错了
  - 解决方案： 修改代码
