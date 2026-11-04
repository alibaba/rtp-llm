#pragma once

/**
 ** Atomic type.
 **/
typedef struct {
    volatile int counter;
} atomic_t;

typedef struct {
    volatile long long counter;
} atomic64_t;

#ifndef ATOMIC_INIT
#define ATOMIC_INIT(i) {(i)}
#endif

/* Atomic operations. */
#define atomic_read(v) (__atomic_load_n(&(v)->counter, __ATOMIC_RELAXED))
#define atomic_set(v, i) (__atomic_store_n(&(v)->counter, (i), __ATOMIC_RELAXED))

#define atomic_add(i, v) ((void)__sync_add_and_fetch(&(v)->counter, i))
#define atomic_sub(i, v) ((void)__sync_sub_and_fetch(&(v)->counter, i))
#define atomic_sub_and_test(i, v) ((__sync_sub_and_fetch(&(v)->counter, i)))
#define atomic_inc(v) ((void)__sync_fetch_and_add(&(v)->counter, 1))
#define atomic_dec(v) ((void)__sync_fetch_and_sub(&(v)->counter, 1))

#define atomic_add_return(i, v) (__sync_add_and_fetch(&(v)->counter, i))
#define atomic_sub_return(i, v) (__sync_sub_and_fetch(&(v)->counter, i))
#define atomic_inc_return(v) (__sync_add_and_fetch(&(v)->counter, 1))
#define atomic_dec_return(v) (__sync_sub_and_fetch(&(v)->counter, 1))

/* return true if the result is 0, or false otherwise. */
#define atomic_dec_and_test(v) (!(__sync_sub_and_fetch(&(v)->counter, 1)))
#define atomic_inc_and_test(v) (!(__sync_add_and_fetch(&(v)->counter, 1)))

/* general operations */
/* return true if swap is done or false otherwise */
#define compare_and_swap(v, oldval, newval) (__sync_bool_compare_and_swap(&(v)->counter, oldval, newval))
