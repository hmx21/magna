"""Small installed classical function; all quantum gates remain in the file."""


def majority(bits, config):
    """Return strict majority for a nonempty odd-size bit sequence."""
    if not bits or len(bits) % 2 == 0:
        raise ValueError("majority expects an odd positive input count")
    if config:
        raise ValueError("this example has no configurable parameters")
    return (int(2 * sum(bits) > len(bits)),)


def syndrome_recovery(bits, config):
    """Example configured table returning correction bits."""
    from hashlib import sha256
    import json
    leaders=config['leaders']; width=config['width']
    if len(leaders)!=1<<len(bits):raise ValueError('syndrome table dimension')
    if sha256(json.dumps(leaders,separators=(',',':')).encode()).hexdigest()!=config['sha256']:
        raise ValueError('syndrome table hash mismatch')
    correction=leaders[sum(bit<<j for j,bit in enumerate(bits))]
    if type(correction) is not int or correction<0 or correction>>width:raise ValueError('invalid correction')
    return tuple((correction>>j)&1 for j in range(width))
