import time
from smbus2 import SMBus

VL53L0X_DEFAULT_ADDRESS = 0x29

REG_SYSRANGE_START = 0x00
REG_SYSTEM_SEQUENCE_CONFIG = 0x01
REG_ALGO_PHASECAL_CONFIG_TIMEOUT = 0x30
REG_GLOBAL_CONFIG_VCSEL_WIDTH = 0x32
REG_MSRC_CONFIG_CONTROL = 0x60
REG_FINAL_RANGE_CONFIG_MIN_COUNT_RATE = 0x44
REG_FINAL_RANGE_CONFIG_VALID_PHASE_HIGH = 0x48
REG_FINAL_RANGE_CONFIG_VALID_PHASE_LOW = 0x47
REG_MSRC_CONFIG_TIMEOUT_MACROP = 0x46
REG_PRE_RANGE_CONFIG_VCSEL_PERIOD = 0x50
REG_PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x51
REG_PRE_RANGE_CONFIG_VALID_PHASE_HIGH = 0x57
REG_PRE_RANGE_CONFIG_VALID_PHASE_LOW = 0x56
REG_FINAL_RANGE_CONFIG_VCSEL_PERIOD = 0x70
REG_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x71
REG_ALGO_PHASECAL_LIM = 0x40
REG_GPIO_HV_MUX_ACTIVE_HIGH = 0x84
REG_SYSTEM_INTERRUPT_CONFIG_GPIO = 0x0A
REG_SYSTEM_INTERRUPT_CLEAR = 0x0B
REG_RESULT_INTERRUPT_STATUS = 0x13
REG_RESULT_RANGE_STATUS = 0x14
REG_VHV_CONFIG_PAD_SCL_SDA_EXTSUP_HV = 0x89
REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_0 = 0xB0
REG_IDENTIFICATION_MODEL_ID = 0xC0
REG_IDENTIFICATION_REVISION_ID = 0xC2

SEQUENCE_ENABLE_TCC = 0x10
SEQUENCE_ENABLE_DSS = 0x08
SEQUENCE_ENABLE_MSRC = 0x04
SEQUENCE_ENABLE_PRE_RANGE = 0x40
SEQUENCE_ENABLE_FINAL_RANGE = 0x80

MAX_TIMEOUT = 100


def _calc_macro_period(vcsel_period_pclks):
    return (2304 * vcsel_period_pclks * 1655 + 500) // 1000


def _encode_vcsel_period(period_pclks):
    return (period_pclks >> 1) - 1


def _decode_timeout(reg_val):
    return ((reg_val & 0x00FF) << ((reg_val & 0xFF00) >> 8)) + 1


def _encode_timeout(timeout_mclks):
    ls_byte = 0
    ms_byte = 0
    if timeout_mclks > 0:
        ls_byte = timeout_mclks - 1
        while (ls_byte & 0xFFFFFF00) > 0:
            ls_byte >>= 1
            ms_byte += 1
        return (ms_byte << 8) | (ls_byte & 0xFF)
    return 0


def _timeout_mclks_to_us(mclks, vcsel_pclks):
    macro_period_ns = _calc_macro_period(vcsel_pclks)
    return (mclks * macro_period_ns + macro_period_ns // 2) // 1000


def _timeout_us_to_mclks(us, vcsel_pclks):
    macro_period_ns = _calc_macro_period(vcsel_pclks)
    return (us * 1000 + macro_period_ns // 2) // macro_period_ns


I2C_MODE1 = [0x88, 0x00, 0x80, 0x01, 0xFF, 0x01, 0x00, 0x00]
I2C_MODE2 = [0x00, 0x01, 0xFF, 0x00, 0x80, 0x00]

SPAD_INIT0 = [0x80, 0x01, 0xFF, 0x01, 0x00, 0x00, 0xFF, 0x06]
SPAD_INIT1 = [0xFF, 0x07, 0x81, 0x01, 0x80, 0x01, 0x94, 0x6B, 0x83, 0x00]
SPAD_INIT2 = [0xFF, 0x01, 0x00, 0x01, 0xFF, 0x00, 0x80, 0x00]

SPAD_CONFIG = [0xFF, 0x01, 0x4F, 0x00, 0x4E, 0x2C, 0xFF, 0x00, 0xB6, 0xB4]

DEFAULT_TUNING = [
    0xFF, 0x01, 0x00, 0x00, 0xFF, 0x00, 0x09, 0x00, 0x10, 0x00, 0x11,
    0x00, 0x24, 0x01, 0x25, 0xFF, 0x75, 0x00, 0xFF, 0x01, 0x4E, 0x2C,
    0x48, 0x00, 0x30, 0x20, 0xFF, 0x00, 0x30, 0x09, 0x54, 0x00, 0x31,
    0x04, 0x32, 0x03, 0x40, 0x83, 0x46, 0x25, 0x60, 0x00, 0x27, 0x00,
    0x50, 0x06, 0x51, 0x00, 0x52, 0x96, 0x56, 0x08, 0x57, 0x30, 0x61,
    0x00, 0x62, 0x00, 0x64, 0x00, 0x65, 0x00, 0x66, 0xA0, 0xFF, 0x01,
    0x22, 0x32, 0x47, 0x14, 0x49, 0xFF, 0x4A, 0x00, 0xFF, 0x00, 0x7A,
    0x0A, 0x7B, 0x00, 0x78, 0x21, 0xFF, 0x01, 0x23, 0x34, 0x42, 0x00,
    0x44, 0xFF, 0x45, 0x26, 0x46, 0x05, 0x40, 0x40, 0x0E, 0x06, 0x20,
    0x1A, 0x43, 0x40, 0xFF, 0x00, 0x34, 0x03, 0x35, 0x44, 0xFF, 0x01,
    0x31, 0x04, 0x4B, 0x09, 0x4C, 0x05, 0x4D, 0x04, 0xFF, 0x00, 0x44,
    0x00, 0x45, 0x20, 0x47, 0x08, 0x48, 0x28, 0x67, 0x00, 0x70, 0x04,
    0x71, 0x01, 0x72, 0xFE, 0x76, 0x00, 0x77, 0x00, 0xFF, 0x01, 0x0D,
    0x01, 0xFF, 0x00, 0x80, 0x01, 0x01, 0xF8, 0xFF, 0x01, 0x8E, 0x01,
    0x00, 0x01, 0xFF, 0x00, 0x80, 0x00,
]


class VL53L0X:
    def __init__(self, i2c_bus=1, address=VL53L0X_DEFAULT_ADDRESS):
        self.bus_num = i2c_bus
        self.address = address
        self.bus = None
        self.stop_variable = 0
        self.timing_budget = 33000

    def _write_reg(self, reg, val):
        self.bus.write_byte_data(self.address, reg, val)

    def _read_reg(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def _read_reg16(self, reg):
        data = self.bus.read_i2c_block_data(self.address, reg, 2)
        return (data[0] << 8) | data[1]

    def _write_reg16(self, reg, val):
        self.bus.write_i2c_block_data(self.address, reg, [(val >> 8) & 0xFF, val & 0xFF])

    def _write_multi(self, reg, data):
        self.bus.write_i2c_block_data(self.address, reg, data)

    def _read_multi(self, reg, count):
        return self.bus.read_i2c_block_data(self.address, reg, count)

    def _write_reg_list(self, data):
        i = 0
        while i < len(data) - 1:
            self._write_reg(data[i], data[i + 1])
            i += 2

    def _get_spad_info(self):
        self._write_reg_list(SPAD_INIT0)
        self._write_reg(0x83, self._read_reg(0x83) | 0x04)
        self._write_reg_list(SPAD_INIT1)

        timeout = 0
        while timeout < MAX_TIMEOUT:
            if self._read_reg(0x83) != 0x00:
                break
            timeout += 1
            time.sleep(0.005)

        if timeout == MAX_TIMEOUT:
            return None, None

        self._write_reg(0x83, 0x01)
        temp = self._read_reg(0x92)
        count = temp & 0x7F
        is_aperture = (temp & 0x80) != 0

        self._write_reg(0x81, 0x00)
        self._write_reg(0xFF, 0x06)
        self._write_reg(0x83, self._read_reg(0x83) & ~0x04)
        self._write_reg_list(SPAD_INIT2)

        return count, is_aperture

    def _perform_single_ref_calibration(self, vhv_init):
        self._write_reg(REG_SYSRANGE_START, 0x01 | vhv_init)
        timeout = 0
        while (self._read_reg(REG_RESULT_INTERRUPT_STATUS) & 0x07) == 0:
            timeout += 1
            time.sleep(0.005)
            if timeout > MAX_TIMEOUT:
                return False
        self._write_reg(REG_SYSTEM_INTERRUPT_CLEAR, 0x01)
        self._write_reg(REG_SYSRANGE_START, 0x00)
        return True

    def _get_sequence_step_timeouts(self, enables):
        pre_range_vcsel = ((self._read_reg(REG_PRE_RANGE_CONFIG_VCSEL_PERIOD) + 1) << 1)
        msrc_dss_tcc_mclks = self._read_reg(REG_MSRC_CONFIG_TIMEOUT_MACROP) + 1
        msrc_dss_tcc_us = _timeout_mclks_to_us(msrc_dss_tcc_mclks, pre_range_vcsel)

        pre_range_mclks = _decode_timeout(self._read_reg16(REG_PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI))
        pre_range_us = _timeout_mclks_to_us(pre_range_mclks, pre_range_vcsel)

        final_range_vcsel = ((self._read_reg(REG_FINAL_RANGE_CONFIG_VCSEL_PERIOD) + 1) << 1)
        final_range_mclks = _decode_timeout(self._read_reg16(REG_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI))

        if enables & SEQUENCE_ENABLE_PRE_RANGE:
            final_range_mclks -= pre_range_mclks

        final_range_us = _timeout_mclks_to_us(final_range_mclks, final_range_vcsel)

        return {
            'pre_range_vcsel': pre_range_vcsel,
            'final_range_vcsel': final_range_vcsel,
            'msrc_dss_tcc_mclks': msrc_dss_tcc_mclks,
            'msrc_dss_tcc_us': msrc_dss_tcc_us,
            'pre_range_mclks': pre_range_mclks,
            'pre_range_us': pre_range_us,
            'final_range_mclks': final_range_mclks,
            'final_range_us': final_range_us,
        }

    def _set_measurement_timing_budget(self, budget_us):
        if budget_us < 20000:
            return False

        enables = self._read_reg(REG_SYSTEM_SEQUENCE_CONFIG)
        timeouts = self._get_sequence_step_timeouts(enables)

        used = 1320 + 960

        if enables & SEQUENCE_ENABLE_TCC:
            used += timeouts['msrc_dss_tcc_us'] + 590
        if enables & SEQUENCE_ENABLE_DSS:
            used += 2 * (timeouts['msrc_dss_tcc_us'] + 690)
        elif enables & SEQUENCE_ENABLE_MSRC:
            used += timeouts['msrc_dss_tcc_us'] + 660
        if enables & SEQUENCE_ENABLE_PRE_RANGE:
            used += timeouts['pre_range_us'] + 660
        if enables & SEQUENCE_ENABLE_FINAL_RANGE:
            used += 550
            if used > budget_us:
                return False
            final_us = budget_us - used
            final_mclks = _timeout_us_to_mclks(final_us, timeouts['final_range_vcsel'])
            if enables & SEQUENCE_ENABLE_PRE_RANGE:
                final_mclks += timeouts['pre_range_mclks']
            self._write_reg16(REG_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI, _encode_timeout(final_mclks))
            self.timing_budget = budget_us

        return True

    def _get_measurement_timing_budget(self):
        enables = self._read_reg(REG_SYSTEM_SEQUENCE_CONFIG)
        timeouts = self._get_sequence_step_timeouts(enables)

        budget = 1910 + 960
        if enables & SEQUENCE_ENABLE_TCC:
            budget += timeouts['msrc_dss_tcc_us'] + 590
        if enables & SEQUENCE_ENABLE_DSS:
            budget += 2 * (timeouts['msrc_dss_tcc_us'] + 690)
        elif enables & SEQUENCE_ENABLE_MSRC:
            budget += timeouts['msrc_dss_tcc_us'] + 660
        if enables & SEQUENCE_ENABLE_PRE_RANGE:
            budget += timeouts['pre_range_us'] + 660
        if enables & SEQUENCE_ENABLE_FINAL_RANGE:
            budget += timeouts['final_range_us'] + 550

        self.timing_budget = budget
        return budget

    def _set_vcsel_pulse_period(self, period_type, period_pclks):
        vcsel_reg = _encode_vcsel_period(period_pclks)
        enables = self._read_reg(REG_SYSTEM_SEQUENCE_CONFIG)
        timeouts = self._get_sequence_step_timeouts(enables)

        if period_type == 0:  # pre-range
            phase_high_map = {12: 0x18, 14: 0x30, 16: 0x40, 18: 0x50}
            if period_pclks not in phase_high_map:
                return False
            self._write_reg(REG_PRE_RANGE_CONFIG_VALID_PHASE_HIGH, phase_high_map[period_pclks])
            self._write_reg(REG_PRE_RANGE_CONFIG_VALID_PHASE_LOW, 0x08)
            self._write_reg(REG_PRE_RANGE_CONFIG_VCSEL_PERIOD, vcsel_reg)
            new_mclks = _timeout_us_to_mclks(timeouts['pre_range_us'], period_pclks)
            self._write_reg16(REG_PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI, _encode_timeout(new_mclks))
            new_mclks = _timeout_us_to_mclks(timeouts['msrc_dss_tcc_us'], period_pclks)
            self._write_reg(REG_MSRC_CONFIG_TIMEOUT_MACROP, min(new_mclks - 1, 255))
        else:  # final-range
            fr_map = {
                8: (0x10, 0x08, 0x02, 0x0C, 0x30),
                10: (0x28, 0x08, 0x03, 0x09, 0x20),
                12: (0x38, 0x08, 0x03, 0x08, 0x20),
                14: (0x48, 0x08, 0x03, 0x07, 0x20),
            }
            if period_pclks not in fr_map:
                return False
            ph, pl, w, t, lim = fr_map[period_pclks]
            self._write_reg(REG_FINAL_RANGE_CONFIG_VALID_PHASE_HIGH, ph)
            self._write_reg(REG_FINAL_RANGE_CONFIG_VALID_PHASE_LOW, pl)
            self._write_reg(REG_GLOBAL_CONFIG_VCSEL_WIDTH, w)
            self._write_reg(REG_ALGO_PHASECAL_CONFIG_TIMEOUT, t)
            self._write_reg(0xFF, 0x01)
            self._write_reg(REG_ALGO_PHASECAL_LIM, lim)
            self._write_reg(0xFF, 0x00)
            self._write_reg(REG_FINAL_RANGE_CONFIG_VCSEL_PERIOD, vcsel_reg)
            new_mclks = _timeout_us_to_mclks(timeouts['final_range_us'], period_pclks)
            if enables & SEQUENCE_ENABLE_PRE_RANGE:
                new_mclks += timeouts['pre_range_mclks']
            self._write_reg16(REG_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI, _encode_timeout(new_mclks))

        self._set_measurement_timing_budget(self.timing_budget)

        seq = self._read_reg(REG_SYSTEM_SEQUENCE_CONFIG)
        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, 0x02)
        self._perform_single_ref_calibration(0x00)
        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, seq)

        return True

    def _read_range_continuous_mm(self):
        timeout = 0
        while (self._read_reg(REG_RESULT_INTERRUPT_STATUS) & 0x07) == 0:
            timeout += 1
            time.sleep(0.005)
            if timeout > MAX_TIMEOUT:
                return 8190
        range_val = self._read_reg16(REG_RESULT_RANGE_STATUS + 10)
        self._write_reg(REG_SYSTEM_INTERRUPT_CLEAR, 0x01)
        return range_val

    def open(self):
        self.bus = SMBus(self.bus_num)
        model = self._read_reg(REG_IDENTIFICATION_MODEL_ID)
        if model != 0xEE:
            self.bus.close()
            self.bus = None
            raise RuntimeError(f"VL53L0X not found (model ID: 0x{model:02X}, expected 0xEE)")

        self._write_reg(REG_VHV_CONFIG_PAD_SCL_SDA_EXTSUP_HV,
                        self._read_reg(REG_VHV_CONFIG_PAD_SCL_SDA_EXTSUP_HV) | 0x01)
        self._write_reg_list(I2C_MODE1)
        self.stop_variable = self._read_reg(0x91)
        self._write_reg_list(I2C_MODE2)
        self._write_reg(REG_SYSRANGE_START, 0x00)
        self._write_reg(REG_MSRC_CONFIG_CONTROL, self._read_reg(REG_MSRC_CONFIG_CONTROL) | 0x12)
        self._write_reg16(REG_FINAL_RANGE_CONFIG_MIN_COUNT_RATE, 32)
        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, 0xFF)

        spad_count, is_aperture = self._get_spad_info()
        if spad_count is None:
            self.bus.close()
            self.bus = None
            raise RuntimeError("Failed to get SPAD info")

        ref_spad_map = self._read_multi(REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_0, 6)
        self._write_reg_list(SPAD_CONFIG)

        first_spad = 12 if is_aperture else 0
        spads_enabled = 0
        for i in range(48):
            if i < first_spad or spads_enabled == spad_count:
                ref_spad_map[i >> 3] &= ~(1 << (i & 7))
            elif ref_spad_map[i >> 3] & (1 << (i & 7)):
                spads_enabled += 1

        self._write_multi(REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_0, ref_spad_map)
        self._write_reg_list(DEFAULT_TUNING)

        self._write_reg(REG_SYSTEM_INTERRUPT_CONFIG_GPIO, 0x04)
        self._write_reg(REG_GPIO_HV_MUX_ACTIVE_HIGH,
                        self._read_reg(REG_GPIO_HV_MUX_ACTIVE_HIGH) & ~0x10)
        self._write_reg(REG_SYSTEM_INTERRUPT_CLEAR, 0x01)
        self.timing_budget = self._get_measurement_timing_budget()
        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, 0xE8)
        self._set_measurement_timing_budget(self.timing_budget)

        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, 0x01)
        if not self._perform_single_ref_calibration(0x40):
            self.bus.close()
            self.bus = None
            raise RuntimeError("VHV calibration failed")

        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, 0x02)
        if not self._perform_single_ref_calibration(0x00):
            self.bus.close()
            self.bus = None
            raise RuntimeError("Phase calibration failed")

        self._write_reg(REG_SYSTEM_SEQUENCE_CONFIG, 0xE8)

    def close(self):
        if self.bus:
            self.bus.close()
            self.bus = None

    def read_single(self):
        self._write_reg(0x80, 0x01)
        self._write_reg(0xFF, 0x01)
        self._write_reg(0x00, 0x00)
        self._write_reg(0x91, self.stop_variable)
        self._write_reg(0x00, 0x01)
        self._write_reg(0xFF, 0x00)
        self._write_reg(0x80, 0x00)

        self._write_reg(REG_SYSRANGE_START, 0x01)

        timeout = 0
        while self._read_reg(REG_SYSRANGE_START) & 0x01:
            timeout += 1
            time.sleep(0.005)
            if timeout > MAX_TIMEOUT:
                return None

        return self._read_range_continuous_mm()

    def start_continuous(self):
        self._write_reg(0x80, 0x01)
        self._write_reg(0xFF, 0x01)
        self._write_reg(0x00, 0x00)
        self._write_reg(0x91, self.stop_variable)
        self._write_reg(0x00, 0x01)
        self._write_reg(0xFF, 0x00)
        self._write_reg(0x80, 0x00)
        self._write_reg(REG_SYSRANGE_START, 0x02)

    def read_continuous(self):
        if (self._read_reg(REG_RESULT_INTERRUPT_STATUS) & 0x07) == 0:
            return None
        distance = self._read_reg16(REG_RESULT_RANGE_STATUS + 10)
        self._write_reg(REG_SYSTEM_INTERRUPT_CLEAR, 0x01)
        return distance

    def stop_continuous(self):
        self._write_reg(REG_SYSRANGE_START, 0x01)
        self._write_reg(0xFF, 0x01)
        self._write_reg(0x00, 0x00)
        self._write_reg(0x91, self.stop_variable)
        self._write_reg(0x00, 0x01)
        self._write_reg(0xFF, 0x00)

    def set_long_range(self):
        self._write_reg16(REG_FINAL_RANGE_CONFIG_MIN_COUNT_RATE, 13)
        self._set_vcsel_pulse_period(0, 18)
        self._set_vcsel_pulse_period(1, 14)

    def get_model(self):
        model = self._read_reg(REG_IDENTIFICATION_MODEL_ID)
        revision = self._read_reg(REG_IDENTIFICATION_REVISION_ID)
        return model, revision
