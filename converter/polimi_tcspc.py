"""
Functions to decode POLIMI-TCSPC timetag data.
"""

import numpy as np


def is_polimi_tcspc(filename):
    with open(filename, 'rb') as f:
        HEADER_SIZE = 48
        header = f.read(HEADER_SIZE)

    # Extract the initial 48-bytes string:
    try:
        label_num_bytes = header.decode().find('\r\n')
    except UnicodeDecodeError:
        valid = False
    else:
        label = header[:label_num_bytes]
        valid =  label == b'POLIMI TCSPC-16CH v001'
    return valid


def get_fifo_full_array(filename):
    buffer, header_info = _read_raw_data(filename, header=True, raise_filesize=False)
    dtype_ch = np.dtype([('ch_byte', 'u1'), ('a', 'u1', 3), ('board_byte', 'u1'), ('b', 'u1')])
    data_ch = np.frombuffer(buffer, dtype_ch)
    overflow = np.right_shift(data_ch['board_byte'], 7)
    return overflow


def _read_raw_data(filename, header=True, raise_filesize=True):
    header_info = {}
    with open(filename, 'rb') as f:
        if header:
            header_info = _read_polimi_tcspc_header(f)
        buffer = f.read()
    
    if np.mod(len(buffer), 6) != 0:
        if raise_filesize:
            raise ValueError('File size is not multiple of 6 bytes.')
        else:
            print('WARNING: File size is not multiple of 6 bytes (%s)' % filename)
            buffer = buffer[:(len(buffer)//6)*6]
    return buffer, header_info
            

def loadfile(filename, header=True, raise_filesize=True):
    buffer, header_info = _read_raw_data(filename, header=header, raise_filesize=raise_filesize)
            
    # dtypes needed to decode the data structure
    dtype_t = np.dtype([('timestamps', '>u4'), ('nanotimes', '>u2')])
    dtype_ch = np.dtype([('ch_byte', 'u1'), ('a', 'u1', 3), ('board_byte', 'u1'), ('b', 'u1')])

    # Buffer view for timestamps and nanotimes fields
    data = np.frombuffer(buffer, dtype_t)

    # Buffer view for bytes with channels (and board) info
    data_ch = np.frombuffer(buffer, dtype_ch)


    ## Extract channels

    # First, extract the **channel bits**, making a copy (`right_shift()` returns a copy)
    channels = np.right_shift(data_ch['ch_byte'], 5)
    assert (channels < 8).all()

    # Then extract the **board** bit
    board = np.bitwise_and(data_ch['board_byte'], 0x40)
    np.right_shift(board, 6, out=board)
    assert (board <= 1).all()

    # Finally use the the board bit as 4th channel-bit
    np.bitwise_or(channels, np.left_shift(board, 3), out=channels)


    ## Extract timestamps

    # To obtain the correct timestamps we need to set to 0 the 3 MSB of the
    # byte containing the channel info. For efficiency we do this operation
    # in-place. No data is lost because the channel data has been already copied
    data_ch.setflags(write=True)
    np.bitwise_and(data_ch['ch_byte'], 0x1F, out=data_ch['ch_byte']);
    timestamps = data['timestamps']

    # Correct for rollover
    timestamps_m = _correct_rollover(timestamps, channels)

    # Check that detetctors and timestamps in each ch have same size
    _, counts_ch = np.unique(channels, return_counts=True)
    for t, ch_size in zip(timestamps_m, counts_ch):
        assert t.size == ch_size


    ## Extract FIFO overflow flag
    overflow = np.right_shift(data_ch['board_byte'], 7)
    if overflow.any():
        print('WARNING: Data has gaps (FIFO overflows).')

    ## Extract nanotimes

    # We get the nanotimes by setting the 2 most-significant bits to 0
    nanotimes = data['nanotimes']
    nanotimes.setflags(write=True)
    np.bitwise_and(nanotimes, 0x3FFF, out=nanotimes)
    nanotimes_inverted = (2**14 - 1) - nanotimes

    d = {'timestamps': timestamps_m, 'nanotimes': nanotimes_inverted,
         'channels': channels, 'fifo_full': overflow}
    return d, header_info


def _read_polimi_tcspc_header(fileobj):
    """Read the header and retun a dictionary with header info.
    """
    HEADER_SIZE = 48*48
    header = fileobj.read(HEADER_SIZE)
    header_info = {}

    # Extract the initial 48-bytes string:
    label_num_bytes = header[:48].decode().find('\r\n')
    header_info['label'] = header[:label_num_bytes]
    assert header_info['label'] == b'POLIMI TCSPC-16CH v001'

    # Decode remaining header
    lines = header[48:].decode().replace('\r', '').split('\n')
    iter_lines = iter(lines)
    for line in iter_lines:
        if line.startswith('clock frequency (MHz)'):
            t_clock_Hz = float(line.split('=')[1])*1e6
            header_info['timestamps_clock_Hz'] = t_clock_Hz
        elif line.startswith('channels'):
            num_channels = int(line.split('=')[1])
            header_info['num_channels'] = num_channels
        elif line == 'TAC Bin Sizes (ps):':
            binwidths_line = next(iter_lines)
            tcspc_binwidths_ps = [float(s) for s in binwidths_line.split(',')]
            header_info['tcspc_binwidths_ps'] = tcspc_binwidths_ps
        elif line == 'Decay Shift (ns):':
            offset_line = next(iter_lines)
            tcspc_offset_ns = [float(s) for s in offset_line.split(',')]
            header_info['tcspc_offset_ns'] = tcspc_offset_ns

    return header_info


def _correct_rollover(timestamps, detectors, detectors_range=(0, 15),
                     nbits=29, delta_rollover=1, debug=False):
    """Process 32bit timestamps to correct rollover and sort channels.

    Parameters
    ----------
    timestamps : array (uint32)
        timestamps to be processes (rollover correction and ch separation)
    det : array (int)
        detector number for each timestamp
    nbits : integer
        number of bits used for the timestamps. Default 24.
    delta_rollover : positive integer
        Sets the minimum negative difference between two consecutive timestamps
        that will be recognized as a rollover. Default 1.
    debug : boolean
        enable additional consistency checks and increase verbosity.

    Returns
    -------
    3 lists of arrays (one per ch) for timestamps (int64), big-FIFO full-flags
    (bool) and small-FIFO full flags (bool).
    """
    cumsum, diff = np.cumsum, np.diff
    max_ts = 2**nbits

    if debug :
        assert (detectors >= detectors_range[0]).all()
        assert (detectors <= detectors_range[1]).all()

    timestamps_m = []
    for ch in range(detectors_range[0], detectors_range[1]+1):
        mask = (detectors == ch)
        times32 = timestamps[mask].astype('int32')
        del mask

        if times32.size >= 3:
            # We need at least 2 valid timestamps and the first is invalid
            times64 = np.zeros(times32.size, dtype=np.int64)
            times64[1:] = (diff(times32) < -delta_rollover)
            cumsum(times64, out=times64)
            times64 *= max_ts
            times64 += times32
            del times32
        else:
            # Return an array of size 0 for current ch
            times64 = np.array([], dtype='int64')
        timestamps_m.append(times64)
    return timestamps_m
