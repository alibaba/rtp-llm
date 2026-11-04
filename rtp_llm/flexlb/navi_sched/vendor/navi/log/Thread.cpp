#include <assert.h>
#include <signal.h>

#include "navi/log/Thread.h"


static void *thread_routine(void *pArg)
{
    // Following code will guarantee more predictable latencies as it'll
    // disallow any signal handling in the I/O thread.
    sigset_t signal_set;
    int rc = sigfillset(&signal_set);
    assert(rc == 0);
    rc = pthread_sigmask(SIG_BLOCK, &signal_set, NULL);
    assert(rc == 0);

    navi::Thread *self = (navi::Thread *)pArg;
    self->m_tfn(self->m_pArg);
    (void)rc;

    return NULL;
}

void navi::Thread::start(thread_fn *tfn, void *pArg, const std::string &name)
{
    m_tfn = tfn;
    m_pArg = pArg;
    int rc = pthread_create(&m_threadDescriptor, NULL, thread_routine, this);
    assert(rc == 0);
    (void)(rc);
    if (!name.empty()) {
        pthread_setname_np(m_threadDescriptor, name.c_str());
    }
}

void navi::Thread::stop()
{
    int rc = pthread_join(m_threadDescriptor, NULL);
    assert(rc == 0);
    (void)(rc);
}
