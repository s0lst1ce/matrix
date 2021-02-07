use matrix::{Error, Matrix};

mod matrix_setup {
    use matrix::Matrix;
    use rand::{thread_rng, Rng};

    #[allow(unused)]
    pub fn setup_rand_3x2() -> Matrix<u8, 3, 2> {
        let mut m = [[0; 2]; 3];
        let mut rng = thread_rng();
        m.iter_mut().for_each(|a| rng.fill(a));
        m.into()
    }

    pub fn setup_3x2() -> Matrix<u8, 3, 2> {
        [[1, 2], [3, 4], [5, 6]].into()
    }

    pub fn setup_2x3() -> Matrix<u8, 2, 3> {
        [[9, 8, 7], [6, 5, 4]].into()
    }

    pub fn setup_3x3() -> Matrix<u8, 3, 3> {
        [[1, 2, 1], [3, 4, 1], [1, 5, 6]].into()
    }
}

#[test]
fn add() {
    let mut m = matrix_setup::setup_3x2();
    m += matrix_setup::setup_3x2();
    assert_eq!(m, Matrix::from([[2, 4], [6, 8], [10, 12]]));
}

#[test]
fn mul() {
    let m = matrix_setup::setup_3x2();
    assert_eq!(
        m * matrix_setup::setup_2x3(),
        [[21, 18, 15], [51, 44, 37], [81, 70, 59]].into()
    );
}

#[test]
fn dilate() {
    let mut m = matrix_setup::setup_3x3();
    m.dilate(1, &2u8).unwrap();
    assert_eq!(m, [[1, 2, 1], [6, 8, 2], [1, 5, 6]].into());
}

#[test]
fn dilate_fail_bounds() {
    let mut m = matrix_setup::setup_3x3();
    assert_eq!(m.dilate(4, &1), Err(Error::OutOfBounds));
}

#[test]
fn transvect() {
    let mut m = matrix_setup::setup_3x3();
    m.transvect(0, 1).unwrap();
    assert_eq!(m, [[4, 6, 2], [3, 4, 1], [1, 5, 6]].into())
}

#[test]
fn transvect_fail_bounds() {
    let mut m = matrix_setup::setup_3x3();
    assert_eq!(m.transvect(3, 0), Err(Error::OutOfBounds));
}

#[test]
fn transvect_fail_op() {
    let mut m = matrix_setup::setup_3x3();
    assert_eq!(m.transvect(0, 0), Err(Error::WrongOperation));
}

#[test]
fn permute() {
    let mut m = matrix_setup::setup_3x3();
    m.permute(0, 1).unwrap();
    assert_eq!(m, [[3, 4, 1], [1, 2, 1], [1, 5, 6]].into());
}

#[test]
fn permute_fail_bounds() {
    let mut m = matrix_setup::setup_3x3();
    assert_eq!(m.permute(4, 0), Err(Error::OutOfBounds));
}
