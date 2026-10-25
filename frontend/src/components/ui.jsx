import { Check } from 'lucide-react';
import './ui.css';

export function Button({ variant = 'secondary', size = 'md', icon: Icon, children, className = '', ...rest }) {
  return (
    <button type="button" className={['btn', 'btn-' + variant, 'btn-' + size, className].join(' ')} {...rest}>
      {Icon && <Icon size={16} aria-hidden="true" />}
      {children && <span>{children}</span>}
    </button>
  );
}

export function Card({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={'card ' + className}>
      {(title || actions) && (
        <header className="card-header">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      <div className="card-body">{children}</div>
    </section>
  );
}

export function Badge({ tone = 'neutral', children }) {
  return <span className={'badge badge-' + tone}>{children}</span>;
}

export function Stepper({ steps, current }) {
  return (
    <ol className="stepper" aria-label="Progress">
      {steps.map((label, i) => {
        const state = i < current ? 'done' : i === current ? 'current' : 'todo';
        return (
          <li key={label} className={'step step-' + state} aria-current={state === 'current' ? 'step' : undefined}>
            <span className="step-index">{state === 'done' ? <Check size={12} aria-hidden="true" /> : i + 1}</span>
            <span className="step-label">{label}</span>
          </li>
        );
      })}
    </ol>
  );
}

export function Disclosure({ title = 'Technical details', children }) {
  return (
    <details className="disclosure">
      <summary>{title}</summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

export function KeyValue({ items }) {
  return (
    <dl className="kv">
      {items.map((it) => (
        <div key={it.label} className="kv-row">
          <dt>{it.label}</dt>
          <dd className="num">{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Spinner() {
  return <span className="spinner" role="status" aria-label="Loading" />;
}

export function Alert({ tone = 'danger', icon: Icon, title, children, action }) {
  return (
    <div className={'alert alert-' + tone} role={tone === 'danger' ? 'alert' : 'status'}>
      {Icon && <Icon size={16} aria-hidden="true" className="alert-icon" />}
      <div className="alert-body">
        {title && <strong>{title}</strong>}
        {children && <div>{children}</div>}
      </div>
      {action}
    </div>
  );
}
