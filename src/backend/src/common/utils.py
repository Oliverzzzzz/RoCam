import os
import netifaces

def set_scheduler_fifo(priority: int):
    """
    Use FIFO real-time scheduling for the current thread.
    
    Priority: 1-99 (99 is highest, preempts all non-RT processes)
    
    WARNING: A FIFO thread that never blocks can freeze the system.
    """
    if not 1 <= priority <= 99:
        raise ValueError("FIFO priority must be between 1 and 99")
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))

def set_scheduler_other(nice_value: int = 0):
    """
    Use normal (CFS) scheduling for the current thread.
    
    Nice: -20 to +19 (-20 is highest priority, 0 is default, +19 is lowest)
    
    This is the default scheduler for most processes. Suitable for
    interactive and general-purpose tasks.
    """
    if not -20 <= nice_value <= 19:
        raise ValueError("Nice value must be between -20 and 19")
    
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    
    # Adjust nice value
    current_nice = os.nice(0)
    adjustment = nice_value - current_nice
    if adjustment != 0:
        os.nice(adjustment)

def set_scheduler_batch(nice_value: int = 19):
    """
    Use batch scheduling for the current thread.
    
    Nice: -20 to +19 (default +19 for background tasks)
    
    Suitable for CPU-intensive non-interactive tasks. Gets longer time
    slices but lower priority than SCHED_OTHER. Still runs before SCHED_IDLE.
    """
    if not -20 <= nice_value <= 19:
        raise ValueError("Nice value must be between -20 and 19")
    
    os.sched_setscheduler(0, os.SCHED_BATCH, os.sched_param(0))
    
    current_nice = os.nice(0)
    adjustment = nice_value - current_nice
    if adjustment != 0:
        os.nice(adjustment)

def ip4_addresses():
    addrs = []
    for iface in netifaces.interfaces():
        info = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in info:
            addrs.append(info[netifaces.AF_INET][0]['addr'])

    return addrs