import { useParams } from 'react-router-dom';
import MainApp from './MainApp';

export default function RunExperimentPage() {
  const { slug } = useParams<{ slug: string }>();
  return (
    <MainApp
      initialMode="experiment"
      experimentSlug={slug}
      autorun
      hideModeToggle
    />
  );
}
