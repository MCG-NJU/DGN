class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, bz=1):
        self.val = val
        self.sum += val
        self.count += bz
        self.avg = self.sum / self.count


def time_format_convert(sec):
    if sec < 60:
        return _process_single_multiple(int(sec),"second")
    elif sec < 3600:
        minutes = int(sec/60)
        seconds = int(sec % 60)
        return _process_single_multiple(minutes,"minute")+_process_single_multiple(seconds,"second")
    elif sec < 3600 * 24:
        hours = int(sec/3600)
        minutes = int((sec - 3600*hours)/60)
        seconds = int(sec%60)
        return _process_single_multiple(hours,"hour")+_process_single_multiple(minutes,"minute")+_process_single_multiple(seconds,"second")
    else:
        days = int(sec/(3600*24))
        left = sec - 3600*24
        hours = int(left / 3600)
        minutes = int((left-3600*hours)/60)
        seconds = int(left % 60)
        return _process_single_multiple(days,"day")+_process_single_multiple(hours,"hour")+_process_single_multiple(minutes,"minute")+_process_single_multiple(seconds,"second")

    
def _process_single_multiple(num, str_):
    if str_ == "second":
        if num == 0 or num == 1:
            return f"{num} "+str_+"."
        else:
            return f"{num} "+str_+"s."
    else:
        if num == 0 or num == 1:
            return f"{num} "+str_+", "
        else:
            return f"{num} "+str_+"s, "