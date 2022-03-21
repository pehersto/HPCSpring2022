#ifndef _INTRIN_WRAPPER_H_
#define _INTRIN_WRAPPER_H_

#include <cstdint>
#include <ostream>
#include <cassert>

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_2__
#include <smmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#if defined(__MIC__)
#include <immintrin.h>
#endif

template <class ValueType> class TypeTraits {
  public:
    static constexpr int SigBits = 0;
};
template <> class TypeTraits<float> {
  public:
    static constexpr int SigBits = 23;
};
template <> class TypeTraits<double> {
  public:
    static constexpr int SigBits = 52;
};

template <class ValueType, int N> class alignas(sizeof(ValueType) * N) Vec {
  public:

    typedef ValueType ScalarType;

    static constexpr int Size() {
      return N;
    }

    static Vec Zero() {
      Vec r;
      for (int i = 0; i < N; i++) r.v[i] = 0;
      return r;
    }

    static Vec Load1(ValueType const* p) {
      Vec r;
      for (int i = 0; i < N; i++) r.v[i] = p[0];
      return r;
    }
    static Vec Load(ValueType const* p) {
      Vec r;
      for (int i = 0; i < N; i++) r.v[i] = p[i];
      return r;
    }
    static Vec LoadAligned(ValueType const* p) {
      Vec r;
      for (int i = 0; i < N; i++) r.v[i] = p[i];
      return r;
    }

    Vec() {}

    Vec(const ValueType& a) {
      for (int i = 0; i < N; i++) v[i] = a;
    }

    void Store(ValueType* p) const {
      for (int i = 0; i < N; i++) p[i] = v[i];
    }
    void StoreAligned(ValueType* p) const {
      for (int i = 0; i < N; i++) p[i] = v[i];
    }

    // Bitwise NOT
    Vec operator~() const {
      Vec r;
      char* vo = (char*)r.v;
      const char* vi = (const char*)this->v;
      for (int i = 0; i < (int)(N*sizeof(ValueType)); i++) vo[i] = ~vi[i];
      return r;
    }

    // Unary plus and minus
    Vec operator+() const {
      return *this;
    }
    Vec operator-() const {
      Vec r;
      for (int i = 0; i < N; i++) r.v[i] = -v[i];
      return r;
    }

    // C-style cast
    template <class RetValueType> explicit operator Vec<RetValueType,N>() const {
      Vec<RetValueType,N> r;
      for (int i = 0; i < N; i++) r.v[i] = (RetValueType)v[i];
      return r;
    }

    // Arithmetic operators
    friend Vec operator*(Vec lhs, const Vec& rhs) {
      for (int i = 0; i < N; i++) lhs.v[i] *= rhs.v[i];
      return lhs;
    }
    friend Vec operator+(Vec lhs, const Vec& rhs) {
      for (int i = 0; i < N; i++) lhs.v[i] += rhs.v[i];
      return lhs;
    }
    friend Vec operator-(Vec lhs, const Vec& rhs) {
      for (int i = 0; i < N; i++) lhs.v[i] -= rhs.v[i];
      return lhs;
    }
    friend Vec FMA(Vec a, const Vec& b, const Vec& c) {
      for (int i = 0; i < N; i++) a.v[i] = a.v[i] * b.v[i] + c.v[i];
      return a;
    }

    // Comparison operators
    friend Vec operator< (Vec lhs, const Vec& rhs) {
      static const ValueType value_zero = const_zero();
      static const ValueType value_one = const_one();
      for (int i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] < rhs.v[i] ? value_one : value_zero);
      return lhs;
    }
    friend Vec operator<=(Vec lhs, const Vec& rhs) {
      static const ValueType value_zero = const_zero();
      static const ValueType value_one = const_one();
      for (int i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] <= rhs.v[i] ? value_one : value_zero);
      return lhs;
    }
    friend Vec operator>=(Vec lhs, const Vec& rhs) {
      static const ValueType value_zero = const_zero();
      static const ValueType value_one = const_one();
      for (int i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] >= rhs.v[i] ? value_one : value_zero);
      return lhs;
    }
    friend Vec operator> (Vec lhs, const Vec& rhs) {
      static const ValueType value_zero = const_zero();
      static const ValueType value_one = const_one();
      for (int i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] > rhs.v[i] ? value_one : value_zero);
      return lhs;
    }
    friend Vec operator==(Vec lhs, const Vec& rhs) {
      static const ValueType value_zero = const_zero();
      static const ValueType value_one = const_one();
      for (int i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] == rhs.v[i] ? value_one : value_zero);
      return lhs;
    }
    friend Vec operator!=(Vec lhs, const Vec& rhs) {
      static const ValueType value_zero = const_zero();
      static const ValueType value_one = const_one();
      for (int i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] != rhs.v[i] ? value_one : value_zero);
      return lhs;
    }

    // Bitwise operators
    friend Vec operator&(Vec lhs, const Vec& rhs) {
      char* vo = (char*)lhs.v;
      const char* vi = (const char*)rhs.v;
      for (int i = 0; i < (int)sizeof(ValueType)*N; i++) vo[i] &= vi[i];
      return lhs;
    }
    friend Vec operator^(Vec lhs, const Vec& rhs) {
      char* vo = (char*)lhs.v;
      const char* vi = (const char*)rhs.v;
      for (int i = 0; i < (int)sizeof(ValueType)*N; i++) vo[i] ^= vi[i];
      return lhs;
    }
    friend Vec operator|(Vec lhs, const Vec& rhs) {
      char* vo = (char*)lhs.v;
      const char* vi = (const char*)rhs.v;
      for (int i = 0; i < (int)sizeof(ValueType)*N; i++) vo[i] |= vi[i];
      return lhs;
    }
    friend Vec AndNot(Vec lhs, const Vec& rhs) {
      return lhs & (~rhs);
    }

    // Assignment operators
    Vec& operator+=(const Vec& rhs) {
      for (int i = 0; i < N; i++) v[i] += rhs.v[i];
      return *this;
    }
    Vec& operator-=(const Vec& rhs) {
      for (int i = 0; i < N; i++) v[i] -= rhs.v[i];
      return *this;
    }
    Vec& operator*=(const Vec& rhs) {
      for (int i = 0; i < N; i++) v[i] *= rhs.v[i];
      return *this;
    }
    Vec& operator&=(const Vec& rhs) {
      char* vo = (char*)this->v;
      const char* vi = (const char*)rhs.v;
      for (int i = 0; i < (int)sizeof(ValueType)*N; i++) vo[i] &= vi[i];
      return *this;
    }
    Vec& operator^=(const Vec& rhs) {
      char* vo = (char*)this->v;
      const char* vi = (const char*)rhs.v;
      for (int i = 0; i < (int)sizeof(ValueType)*N; i++) vo[i] ^= vi[i];
      return *this;
    }
    Vec& operator|=(const Vec& rhs) {
      char* vo = (char*)this->v;
      const char* vi = (const char*)rhs.v;
      for (int i = 0; i < (int)sizeof(ValueType)*N; i++) vo[i] |= vi[i];
      return *this;
    }

    // Other operators
    friend Vec max(Vec lhs, const Vec& rhs) {
      for (int i = 0; i < N; i++) {
        if (lhs.v[i] < rhs.v[i]) lhs.v[i] = rhs.v[i];
      }
      return lhs;
    }
    friend Vec min(Vec lhs, const Vec& rhs) {
      for (int i = 0; i < N; i++) {
        if (lhs.v[i] > rhs.v[i]) lhs.v[i] = rhs.v[i];
      }
      return lhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
      //for (int i = 0; i < (int)sizeof(ValueType)*8; i++) os << ((*(uint64_t*)in.v) & (1UL << i) ? '1' : '0');
      //os << '\n';
      for (int i = 0; i < N; i++) os << in.v[i] << ' ';
      return os;
    }
    friend Vec approx_rsqrt(const Vec& x) {
      Vec r;
      for (int i = 0; i < N; i++) r.v[i] = 1.0 / sqrt(x.v[i]);
      return r;
    }

  private:

    static const ValueType const_zero() {
      union {
        ValueType value;
        unsigned char cvalue[sizeof(ValueType)];
      };
      for (int i = 0; i < (int)sizeof(ValueType); i++) cvalue[i] = 0;
      return value;
    }
    static const ValueType const_one() {
      union {
        ValueType value;
        unsigned char cvalue[sizeof(ValueType)];
      };
      for (int i = 0; i < (int)sizeof(ValueType); i++) cvalue[i] = ~(unsigned char)0;
      return value;
    }

    ValueType v[N];
};

// Other operators
template <class RealVec, class IntVec> RealVec ConvertInt2Real(const IntVec& x) {
  typedef typename RealVec::ScalarType Real;
  typedef typename IntVec::ScalarType Int;
  assert(sizeof(RealVec) == sizeof(IntVec));
  assert(sizeof(Real) == sizeof(Int));
  static constexpr int SigBits = TypeTraits<Real>::SigBits;
  union {
    Int Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
    Real Creal;
  };
  IntVec l(x + IntVec(Cint));
  return *(RealVec*)&l - RealVec(Creal);
}
template <class IntVec, class RealVec> IntVec RoundReal2Int(const RealVec& x) {
  typedef typename RealVec::ScalarType Real;
  typedef typename IntVec::ScalarType Int;
  assert(sizeof(RealVec) == sizeof(IntVec));
  assert(sizeof(Real) == sizeof(Int));
  static constexpr int SigBits = TypeTraits<Real>::SigBits;
  union {
    Int Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
    Real Creal;
  };
  RealVec d(x + RealVec(Creal));
  return *(IntVec*)&d - IntVec(Cint);
}
template <class Vec> Vec RoundReal2Real(const Vec& x) {
  typedef typename Vec::ScalarType Real;
  static constexpr int SigBits = TypeTraits<Real>::SigBits;
  union {
    int64_t Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
    Real Creal;
  };
  Vec Vreal(Creal);
  return (x + Vreal) - Vreal;
}

#ifdef __AVX__
template <> class alignas(sizeof(double)*4) Vec<double,4> {
  typedef __m256d VecType;
  typedef double ValueType;
  static constexpr int N = 4;
  public:

    typedef ValueType ScalarType;

    static constexpr int Size() {
      return N;
    }

    static Vec Zero() {
      Vec r;
      r.v = _mm256_setzero_pd();
      return r;
    }

    static Vec Load1(ValueType const* p) {
      Vec r;
      r.v = _mm256_broadcast_sd(p);
      return r;
    }
    static Vec Load(ValueType const* p) {
      Vec r;
      r.v = _mm256_loadu_pd(p);
      return r;
    }
    static Vec LoadAligned(ValueType const* p) {
      Vec r;
      r.v = _mm256_load_pd(p);
      return r;
    }

    Vec() {}

    Vec(const ValueType& a) {
      v = _mm256_set1_pd(a);
    }

    void Store(ValueType* p) const {
      _mm256_storeu_pd(p, v);
    }
    void StoreAligned(ValueType* p) const {
      _mm256_store_pd(p, v);
    }

    // Bitwise NOT
    Vec operator~() const {
      Vec r;
      static constexpr ScalarType Creal = -1.0;
      r.v = _mm256_xor_pd(v, _mm256_set1_pd(Creal));
      return r;
    }

    // Unary plus and minus
    Vec operator+() const {
      return *this;
    }
    Vec operator-() const {
      return Zero() - (*this);
    }

    // C-style cast
    template <class RetValueType> explicit operator Vec<RetValueType,N>() const {
      Vec<RetValueType,N> r;
      VecType& ret_v = *(VecType*)&r.v;
      ret_v = v;
      return r;
    }

    // Arithmetic operators
    friend Vec operator*(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_mul_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec operator+(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_add_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec operator-(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_sub_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec FMA(Vec a, const Vec& b, const Vec& c) {
      #ifdef __FMA__
      a.v = _mm256_fmadd_pd(a.v, b.v, c.v);
      #else
      a.v = _mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v);
      #endif
      return a;
    }

    // Comparison operators
    friend Vec operator< (Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LT_OS);
      return lhs;
    }
    friend Vec operator<=(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LE_OS);
      return lhs;
    }
    friend Vec operator>=(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GE_OS);
      return lhs;
    }
    friend Vec operator> (Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GT_OS);
      return lhs;
    }
    friend Vec operator==(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_EQ_OS);
      return lhs;
      return lhs;
    }
    friend Vec operator!=(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NEQ_OS);
      return lhs;
    }

    // Bitwise operators
    friend Vec operator&(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_and_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec operator^(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_xor_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec operator|(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_or_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec AndNot(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_andnot_pd(rhs.v, lhs.v);
      return lhs;
    }

    // Assignment operators
    Vec& operator*=(const Vec& rhs) {
      v = _mm256_mul_pd(v, rhs.v);
      return *this;
    }
    Vec& operator+=(const Vec& rhs) {
      v = _mm256_add_pd(v, rhs.v);
      return *this;
    }
    Vec& operator-=(const Vec& rhs) {
      v = _mm256_sub_pd(v, rhs.v);
      return *this;
    }
    Vec& operator&=(const Vec& rhs) {
      v = _mm256_and_pd(v, rhs.v);
      return *this;
    }
    Vec& operator^=(const Vec& rhs) {
      v = _mm256_xor_pd(v, rhs.v);
      return *this;
    }
    Vec& operator|=(const Vec& rhs) {
      v = _mm256_or_pd(v, rhs.v);
      return *this;
    }

    // Other operators
    friend Vec max(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_max_pd(lhs.v, rhs.v);
      return lhs;
    }
    friend Vec min(Vec lhs, const Vec& rhs) {
      lhs.v = _mm256_min_pd(lhs.v, rhs.v);
      return lhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
      union {
        VecType vec;
        ValueType val[N];
      };
      vec = in.v;
      for (int i = 0; i < N; i++) os << val[i] << ' ';
      return os;
    }
    friend Vec approx_rsqrt(const Vec& x) {
      Vec r;
      r.v = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x.v)));
      return r;
    }

    template <class Vec> friend Vec RoundReal2Real(const Vec& x);
    template <class Vec> friend void sincos_intrin(Vec& sinx, Vec& cosx, const Vec& x);

  private:

    VecType v;
};

template <> inline Vec<double,4> RoundReal2Real(const Vec<double,4>& x) {
  Vec<double,4> r;
  r.v = _mm256_round_pd(x.v,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
  return r;
}
#elif defined(__SSE4_2__)
template <> class alignas(sizeof(double)*4) Vec<double,4> {
  //typedef __m128d[2] VecType;
  typedef struct VecType { __m128d x[2]; } VecType;
  __m128d v[2];

  typedef double ValueType;
  static constexpr int N = 4;
  public:

    typedef ValueType ScalarType;

    static constexpr int Size() {
      return N;
    }

    static Vec Zero() {
      Vec r;
      r.v[0] = _mm_setzero_pd();
      r.v[1] = _mm_setzero_pd();
      return r;
    }

    static Vec Load1(ValueType const* p) {
      Vec r;
      r.v[0] = _mm_load1_pd(p);
      r.v[1] = _mm_load1_pd(p);
      return r;
    }
    static Vec Load(ValueType const* p) {
      Vec r;
      r.v[0] = _mm_loadu_pd(p+0);
      r.v[1] = _mm_loadu_pd(p+2);
      return r;
    }
    static Vec LoadAligned(ValueType const* p) {
      Vec r;
      r.v[0] = _mm_load_pd(p+0);
      r.v[1] = _mm_load_pd(p+2);
      return r;
    }

    Vec() {}

    Vec(const ValueType& a) {
      v[0] = _mm_set1_pd(a);
      v[1] = _mm_set1_pd(a);
    }

    void Store(ValueType* p) const {
      _mm_storeu_pd(p+0, v[0]);
      _mm_storeu_pd(p+2, v[1]);
    }
    void StoreAligned(ValueType* p) const {
      _mm_store_pd(p+0, v[0]);
      _mm_store_pd(p+2, v[1]);
    }

    // Bitwise NOT
    Vec operator~() const {
      Vec r;
      static constexpr ScalarType Creal = -1.0;
      r.v[0] = _mm_xor_pd(v[0], _mm_set1_pd(Creal));
      r.v[1] = _mm_xor_pd(v[1], _mm_set1_pd(Creal));
      return r;
    }

    // Unary plus and minus
    Vec operator+() const {
      return *this;
    }
    Vec operator-() const {
      return Zero() - (*this);
    }

    // C-style cast
    template <class RetValueType> explicit operator Vec<RetValueType,N>() const {
      Vec<RetValueType,N> r;
      VecType& ret_v = *(VecType*)&r.v;
      ret_v = v;
      return r;
    }

    // Arithmetic operators
    friend Vec operator*(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_mul_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_mul_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator+(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_add_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_add_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator-(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_sub_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_sub_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec FMA(Vec a, const Vec& b, const Vec& c) {
      a.v[0] = _mm_add_pd(_mm_mul_pd(a.v[0], b.v[0]), c.v[0]);
      a.v[1] = _mm_add_pd(_mm_mul_pd(a.v[1], b.v[1]), c.v[1]);
      return a;
    }

    // Comparison operators
    friend Vec operator< (Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_cmplt_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_cmplt_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator<=(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_cmple_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_cmple_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator>=(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_cmpge_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_cmpge_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator> (Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_cmpgt_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_cmpgt_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator==(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_cmpeq_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_cmpeq_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator!=(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_cmpneq_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_cmpneq_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }

    // Bitwise operators
    friend Vec operator&(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_and_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_and_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator^(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_xor_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_xor_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec operator|(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_or_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_or_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec AndNot(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_andnot_pd(rhs.v[0], lhs.v[0]);
      lhs.v[1] = _mm_andnot_pd(rhs.v[1], lhs.v[1]);
      return lhs;
    }

    // Assignment operators
    Vec& operator*=(const Vec& rhs) {
      v[0] = _mm_mul_pd(v[0], rhs.v[0]);
      v[1] = _mm_mul_pd(v[1], rhs.v[1]);
      return *this;
    }
    Vec& operator+=(const Vec& rhs) {
      v[0] = _mm_add_pd(v[0], rhs.v[0]);
      v[1] = _mm_add_pd(v[1], rhs.v[1]);
      return *this;
    }
    Vec& operator-=(const Vec& rhs) {
      v[0] = _mm_sub_pd(v[0], rhs.v[0]);
      v[1] = _mm_sub_pd(v[1], rhs.v[1]);
      return *this;
    }
    Vec& operator&=(const Vec& rhs) {
      v[0] = _mm_and_pd(v[0], rhs.v[0]);
      v[1] = _mm_and_pd(v[1], rhs.v[1]);
      return *this;
    }
    Vec& operator^=(const Vec& rhs) {
      v[0] = _mm_xor_pd(v[0], rhs.v[0]);
      v[1] = _mm_xor_pd(v[1], rhs.v[1]);
      return *this;
    }
    Vec& operator|=(const Vec& rhs) {
      v[0] = _mm_or_pd(v[0], rhs.v[0]);
      v[1] = _mm_or_pd(v[1], rhs.v[1]);
      return *this;
    }

    // Other operators
    friend Vec max(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_max_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_max_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }
    friend Vec min(Vec lhs, const Vec& rhs) {
      lhs.v[0] = _mm_min_pd(lhs.v[0], rhs.v[0]);
      lhs.v[1] = _mm_min_pd(lhs.v[1], rhs.v[1]);
      return lhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
      union {
        VecType vec;
        ValueType val[N];
      };
      vec.x[0] = in.v[0];
      vec.x[1] = in.v[1];
      for (int i = 0; i < N; i++) os << val[i] << ' ';
      return os;
    }
    friend Vec approx_rsqrt(const Vec& x) {
      Vec r;
      r.v[0] = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(x.v[0])));
      r.v[1] = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(x.v[1])));
      return r;
    }

    template <class Vec> friend Vec RoundReal2Real(const Vec& x);
    template <class Vec> friend void sincos_intrin(Vec& sinx, Vec& cosx, const Vec& x);

  private:

};

template <> inline Vec<double,4> RoundReal2Real(const Vec<double,4>& x) {
  Vec<double,4> r;
  r.v[0] = _mm_round_pd(x.v[0],_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
  r.v[1] = _mm_round_pd(x.v[1],_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
  return r;
}
#endif

#endif
