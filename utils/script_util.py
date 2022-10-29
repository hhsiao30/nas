def write_sdc(ref_sdc, new_sdc, freq, design):
    period = 1./freq
    with open(new_sdc, 'w') as new_file:
        with open(ref_sdc, 'r') as ref_file:
            for line in ref_file.readlines():
                line = line.strip('\n')
                line = line.strip()
                if (len(line) == 0):
                    continue
                if line.startswith("create_clock"):
                    splits = line.split()
                    if design in {"ldpc", "aes"}:
                        splits[4] = str(period)
                        splits[8] = str(period/2)
                    elif design == "vga":
                        splits[6] = str(period)
                        splits[10] = str(period/2)
                    newline = ' '.join(splits)
                    new_file.write(newline)
                else:
                    new_file.write(line)
                new_file.write('\n')