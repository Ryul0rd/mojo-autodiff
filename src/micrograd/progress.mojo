

@value
struct ProgressBar:
    var current: Int
    var total: Int
    var desc: String
    var ncols: Int

    fn __init__(inout self, total: Int, desc: String='', ncols: Int=120):
        self.current = 0
        self.total = total
        self.desc = desc
        self.ncols = ncols
        self.draw()

    fn draw(self):
        var bar_head = String('')
        var space_remaining = self.ncols

        if len(self.desc) > 0:
            var desc = self.desc + ': '
            bar_head += desc
            space_remaining -= len(desc)
        var percentage = self.left_pad(100 * self.current // self.total, 3) + '%|'
        bar_head += percentage
        space_remaining -= len(percentage)
        var bar_tail = String('| ') + self.left_pad(self.current, self.n_digits(self.total)) + '/' + String(self.total)
        space_remaining -= len(bar_tail)

        var trunk_size = space_remaining
        var filled = trunk_size * self.current / self.total
        var fully_filled = int(filled)
        var bar_trunk = String('#') * fully_filled + String(' ') * (trunk_size - fully_filled)

        print(bar_head + bar_trunk + bar_tail)

    fn update(inout self, increment: Int=1):
        self.current += increment
        self.remove_last_line()
        self.draw()

    fn remove_last_line(self):
        print('\x1b[1A\x1b[2K', end='')

    fn n_digits(self, owned int: Int) -> Int:
        var n = 1
        while int >= 10:
            n += 1
            int = int // 10
        return n

    fn left_pad(self, s: String, pad_to: Int) -> String:
        var spaces_needed = pad_to - len(s)
        if spaces_needed <= 0:
            return s
        return String(' ') * spaces_needed + s
